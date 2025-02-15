

// Scala Project to compute fake vs legitimate review using gaussian mixture model.
// Group Member:
// Arihant
// Anish
// Pawan
// Shobhit


import org.apache.spark.sql.types.{StructField, IntegerType, DoubleType, StructType}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Column, ColumnName, DataFrame, Row, SaveMode, SparkSession, types}
import org.apache.spark.ml.feature.MaxAbsScaler
import org.apache.spark.ml.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.evaluation._



object ProjectHandler {
  //This is the main function that runs on execution

  def main(args: Array[ String ] )  : Unit =  {
    //We are adding more timeout to battle some data frame operations on the server
    val sparkConf = new SparkConf().setAppName("FakeReviewsClassification").set("spark.sql.broadcastTimeout", "36000")
    val sparkContext = new SparkContext( sparkConf )
    val Spark = SparkSession.builder.config( conf = sparkConf ).appName("FakeReviewsClassification").getOrCreate()

    import  Spark.implicits._

    if (args.length < 4) {
      println("Usage:  Input Output RowLimit PrintFlag")
      System.exit( 1 )
    }

    val inputFilePathForData = args(0)
    val outputFilePathForCSV = args(1)

    var printFlag = true
    if (args(3) == 0) {
      printFlag = false
    }


    val dataFrameFromParquet = Spark.read.parquet(inputFilePathForData)

    dataFrameFromParquet.createOrReplaceTempView("parquetFile")
    var query = "SELECT * FROM parquetFile LIMIT "+ args(2)
    val originalDataFrame = Spark.sql(query)


    val uniqueIdGenerator = udf((product_id: String, customer_id: String, review_date: String) => {
      product_id + "_" + customer_id + "_" + review_date
    })

    val computedDataFrameA = originalDataFrame.withColumn("review_id", uniqueIdGenerator($"product_id", $"customer_id", $"review_date"))
    computedDataFrameA.cache()

    // computing sentiment

    val reviewsSummaryDataFrame = computedDataFrameA.select("review_id", "review_body")

    def sentimentAnalysis: (String => Int) = { input => SentimentAnalyzer.mainSentiment(input) }

    val sentimentAnalysisUDF = udf(sentimentAnalysis)


    val computedDataFrameB = computedDataFrameA.select("review_id", "product_id", "helpful_votes", "star_rating", "customer_id", "review_date", "review_body")

    computedDataFrameB.cache()
    if (printFlag) {
      computedDataFrameB.show()
    }


    // Gnerating sentiment column.

    val computedDataFrameC = computedDataFrameB.withColumn("sentiment", sentimentAnalysisUDF(reviewsSummaryDataFrame("review_body"))).cache.drop("review_body")
    computedDataFrameC.cache()
    if (printFlag) {
      computedDataFrameC.show()
    }


    // Find average sentiment score  and average rating for each product

    val averageSentimentRatingDataFrame = computedDataFrameC.select("product_id", "sentiment", "star_rating")
    val productIdSentimentMap = averageSentimentRatingDataFrame.columns.map((_ -> "mean")).toMap
    val averageSentimentRatingDataFrameA = averageSentimentRatingDataFrame.groupBy("product_id").agg(productIdSentimentMap)
    averageSentimentRatingDataFrameA.cache()

    val averageSentimentRatingDataFrameB = averageSentimentRatingDataFrameA.drop("avg(product_id)")


    val computedDataFrameE = computedDataFrameC.join(averageSentimentRatingDataFrameB, Seq("product_id"))

    // Function to compute the distance of each datapoint from its mean.S
    def meanDistance(mu: Double, data: Double): Double = {
      math.abs(mu - data)
    }

    def meanDistanceUDF = udf(meanDistance _)

    val computedDataFrameF = computedDataFrameE.withColumn("sentimentDelta", meanDistanceUDF(computedDataFrameE("avg(sentiment)"), computedDataFrameE("sentiment")))
    val computedDataFrameG = computedDataFrameF.withColumn("overallDelta", meanDistanceUDF(computedDataFrameF("avg(star_rating)"), computedDataFrameF("star_rating")))


    // Generate feature vector
    val model_features = new VectorAssembler()
      .setInputCols( Array ("overallDelta", "sentimentDelta", "helpful_votes")).setOutputCol("features")

    val attributesDataFrame = model_features.transform (computedDataFrameG)


    // Min Max Standardization
    val Standardizer = new MaxAbsScaler().setInputCol("features").setOutputCol("standardizedfeatures")

    val Standardizer_model = Standardizer. fit(attributesDataFrame)

    var transformedData = Standardizer_model. transform(attributesDataFrame)



    // custom csvSchema defined for csv
    var outlineType = types.StructType( StructField("K", IntegerType , false ) :: StructField(" s_width", DoubleType , false ) :: Nil)

    var clusterSilhouetteDataFrame = Spark.createDataFrame (sparkContext.emptyRDD[Row], outlineType )

    def getModel(k:Int,transformedData:DataFrame):GaussianMixtureModel={
      val gausian_mixture_model = new GaussianMixture()
        .setK(k).setFeaturesCol("standardizedfeatures").setMaxIter(100)

      val gmm = gausian_mixture_model.fit(transformedData)
      return gmm
    }

    def writeCSV(dataframe:DataFrame,path:String)={
      dataframe.coalesce(1).write.csv(path)
    }
    // Compute silhouette width value against K clusters ranging from 2 to 51
    for (k <- 2 to 51) {

      var  gmm =  getModel(k,transformedData)
      val estimated_value = gmm.transform(transformedData)

      val model_cosine = new ClusteringEvaluator() .
        setDistanceMeasure ("cosine")
      val s_width = model_cosine.evaluate(estimated_value)

      if (printFlag) {
        println("silhouette width " + s_width + " for K " + k)
      }


      val nextLine = Seq((k, s_width)).toDF("cluster", "s_width")

      clusterSilhouetteDataFrame = clusterSilhouetteDataFrame.union(nextLine)


      val checkNormalDistributionConfidence: Any => Boolean = _ .asInstanceOf[ Vector ].toArray.exists(x =>x >  0.90)

      val checkNormalDistributionConfidenceUdf = udf(checkNormalDistributionConfidence)

      val reviewerDataFrame = estimated_value.withColumn("isNormal", checkNormalDistributionConfidenceUdf($"probability"))


      val reviewerDataFrameB = reviewerDataFrame.columns.foldLeft(reviewerDataFrame)((present, p) => present.withColumn(p, col(p).cast("String")))
      val reviewerDataFrameC = reviewerDataFrameB.select("review_id", "product_id", "customer_id", "prediction", "isNormal")


      writeCSV(reviewerDataFrameC,outputFilePathForCSV + "_" + k)

      if (printFlag) {
        clusterSilhouetteDataFrame.show()
      }

    }
    writeCSV(clusterSilhouetteDataFrame,outputFilePathForCSV + "_silhouette_scoreVScluster")
  }

}
