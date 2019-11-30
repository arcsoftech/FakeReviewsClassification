

// Scala Project to compute fake vs legitimate review using gaussian mixture model.
// Group Member:
// Arihant
// Anish
// Pawan
// Shobhit


import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{ Row, Column, ColumnName,SaveMode, SparkSession, types}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.clustering.GaussianMixture
import org.apache.spark.ml.evaluation._



object ProjectHandler {
  def main(args: Array[String]) : Unit = {
    val sparkConf = new SparkConf().setAppName("FakeReviewsClassification").set("spark.sql.broadcastTimeout", "36000"); //AWS
    val sc = new SparkContext(sparkConf)
    val Spark = SparkSession.builder
      .config(conf = sparkConf)
      .appName("FakeReviewsClassification")
      .getOrCreate()

    import Spark.implicits._

    if (args.length < 4) {
      println("Usage:  Input Output RowLimit PrintFlag")
      System.exit(1)
    }

    val input = args(0)
    val output = args(1)
    var printFlag = true
    if (args(3) == 0) {
      printFlag = false
    }


val parquetFileDF = Spark.read.parquet(input)

// Parquet files can also be used to create a temporary view and then used in SQL statements
parquetFileDF.createOrReplaceTempView("parquetFile")
var query = "SELECT * FROM parquetFile LIMIT "+ args(2)
val original_df = Spark.sql(query)


    val uniqueIdGenerator = udf((product_id: String, customer_id: String, review_date: String) => {
      product_id + "_" + customer_id + "_" + review_date
    })

    val computedDataFrameA = original_df.withColumn("review_id", uniqueIdGenerator($"product_id", $"customer_id", $"review_date"))
    computedDataFrameA.cache()

    // computing sentiment

    val reviewsSummaryDataFrame = computedDataFrameA.select("review_id", "review_body")

    def sentimentAnalysis: (String => Int) = { s => SentimentAnalyzer.mainSentiment(s) }

    val sentimentAnalysisUDF = udf(sentimentAnalysis)


    val computedDataFrameB = computedDataFrameA.select("review_id", "product_id", "helpful_votes", "star_rating", "customer_id", "review_date", "review_body");

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
    val averageSentimentRatingDataFrameA = averageSentimentRatingDataFrame.groupBy("product_id").agg(productIdSentimentMap);
    averageSentimentRatingDataFrameA.cache()
    if (printFlag) {
      averageSentimentRatingDataFrameA.show()
    }


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
    val feature_assembler = new VectorAssembler()
      .setInputCols(Array("overallDelta", "sentimentDelta", "helpful_votes"))
      .setOutputCol("features")

    val featuresDF = feature_assembler.transform(computedDataFrameG)
    if (printFlag) {
      println("Feature combined using VectorAssembler")
      featuresDF.show()
    }


    // Min Max Standardization
    val minMaxStandardizer = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("standardizedfeatures")

    val minMaxStandardizer_model = minMaxStandardizer.fit(featuresDF)

    val transformedData = minMaxStandardizer_model.transform(featuresDF)

    if (printFlag) {
      println(s"Features scaled to range: [${minMaxStandardizer.getMin}, ${minMaxStandardizer.getMax}]")
      transformedData.show()
    }


    // custom csvSchema defined for csv
    var Schema = types.StructType(
      StructField("K", IntegerType, false) ::
        StructField("s_width", DoubleType, false) :: Nil)

    var clusterSilhouette_df = Spark.createDataFrame(sc.emptyRDD[Row], Schema)


    // Compute silhouette width value against K clusters ranging from 2 to 51
    for (k <- 2 to 51) {

      val gausian_mixture_model = new GaussianMixture()
        .setK(k).setFeaturesCol("standardizedfeatures").setMaxIter(100)

      val gmm = gausian_mixture_model.fit(transformedData)
      val estimated_value = gmm.transform(transformedData)

      val model_evaluator = new ClusteringEvaluator().setDistanceMeasure("cosine")
      val s_width = model_evaluator.evaluate(estimated_value);
      println("silhouette width " + s_width + " for K " + k);

      val nextLine = Seq((k, s_width)).toDF("cluster", "s_width");
      clusterSilhouette_df = clusterSilhouette_df.union(nextLine)


      for (i <- 0 until gmm.getK) {
        println(s"Gaussian $i:\nweight=${gmm.weights(i)}\n" +
          s"mu=${gmm.gaussians(i).mean}\nsigma=\n${gmm.gaussians(i).cov}\n")
      }


      val checkNormalDistributionConfidence: Any => Boolean = _.asInstanceOf[DenseVector].toArray.exists(_ > 0.90)
      val checkNormalDistributionConfidenceUdf = udf(checkNormalDistributionConfidence)
      val reviewerDataFrame = estimated_value.withColumn("normal", checkNormalDistributionConfidenceUdf($"probability"))

      if (printFlag) {
        reviewerDataFrame.show();
      }


      val reviewerDataFrameB = reviewerDataFrame.columns.foldLeft(reviewerDataFrame)((current, c) => current.withColumn(c, col(c).cast("String")))
      val reviewerDataFrameC = reviewerDataFrameB.select("review_id", "product_id", "customer_id", "prediction", "normal")

      reviewerDataFrameC.coalesce(1).write.mode(SaveMode.Overwrite).csv(output + "_" + k);

      if (printFlag) {
        clusterSilhouette_df.show()
      }

    }

    clusterSilhouette_df.coalesce(1).write.mode(SaveMode.Overwrite).csv(output + "_silhouette_scoreVScluster");
  }

}
