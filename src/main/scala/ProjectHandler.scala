import SentimentAnalyzer.extractSentiment
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{Column, ColumnName, Row, SaveMode, SparkSession, types}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.evaluation._
import org.apache.spark.sql.functions._

object ProjectHandler {
  def main(args: Array[String]): Unit = {

    Logger.getRootLogger.setLevel(Level.WARN)

    val sparkConf = new SparkConf().setAppName("FakeReviewsClassification").set("spark.sql.broadcastTimeout","36000");                       //AWS

    val sc = new SparkContext(sparkConf)

    val sparkSession = SparkSession.builder
      .config(conf = sparkConf)
      .appName("FakeReviewsClassification")
      .getOrCreate()

    sc.setLogLevel("ERROR")
    if (args.length < 2) {
      println("Usage:  InputFilePath OutputFilePath")
      System.exit(1)
    }

    val inputFilePath = args(0)
    val outputFilePath = args(1)



    import sparkSession.implicits._



    val raw_reviews_df = sparkSession.read.option("inferSchema", "true").option("header", "true").csv(inputFilePath)

    val generateCompositeId = udf( (first: String, second: String, third: String) => { first + "_" + second + "_" + third } )


    def uniqueIdGenerator(productId: ColumnName, customerId:ColumnName, reviewId:ColumnName): Column =  {
      productId + "_" + customerId + "_" + reviewId
    }

    //generating ids for each review
    val reviews_df1  = raw_reviews_df.withColumn("review_id", uniqueIdGenerator($"product_id", $"customer_id", $"review_date"))

    reviews_df1.cache();

    // Extracting Sentiment value for each review
    val reviews_text_df = reviews_df1.select("review_id", "review_body")
def analyzeSentiment: (String => Int) = { s => SentimentAnalyzer.mainSentiment(s) }
 val analyzeSentimentUDF = udf(analyzeSentiment)

    val sentiment_df1 = reviews_text_df.withColumn("sentiment", analyzeSentimentUDF(reviews_text_df("review_body")))
    val sentiment_df2 = sentiment_df1.select("review_id", "sentiment")

    sentiment_df2.cache();

    //Dropping text review column after extracting sentiment
    val reviews_df2 = reviews_df1.select("review_id","product_id","helpful_votes","star_rating","customer_id","review_date");

    reviews_df2.cache()
    reviews_df2.show()


    //Adding calculated sentiment value for each review
    val reviews_df3 = reviews_df2.join(sentiment_df2 ,"review_id")
    reviews_df3.show()
    reviews_df3.cache();



    //calculating average sentiment score for the product
    val product_avg_sentiment_score_df = reviews_df3.select("product_id", "sentiment")
    val asinSentimentMap = product_avg_sentiment_score_df.columns.map((_ -> "mean")).toMap
    val product_avg_sentiment_score_df1 = product_avg_sentiment_score_df.groupBy("product_id").agg(asinSentimentMap);
    product_avg_sentiment_score_df1.show()
    product_avg_sentiment_score_df1.cache()

    val product_avg_sentiment_score_df2 = product_avg_sentiment_score_df1.drop("avg(product_id)")


    println("product_avg_sentiment_score_df2 completed");

    //calculating average overall review score for the product
    val product_avg_overall_df = reviews_df3.select("product_id", "star_rating")
    val asinOverallMap = product_avg_overall_df.columns.map((_ -> "mean")).toMap
    val product_avg_overall_df1 = product_avg_overall_df.groupBy("product_id").agg(asinOverallMap);
    product_avg_overall_df1.show()

    val product_avg_overall_df2 = product_avg_overall_df1.drop("avg(product_id)")

    val reviews_df4 = reviews_df3.join(product_avg_sentiment_score_df2 ,Seq("product_id"))


    val reviews_df5 = reviews_df4.join(product_avg_overall_df2 ,Seq("product_id"))


    //Used to calculate how specific instance is different from group average
    def deltaFunc (avgValue: Double, specificValue:Double) :Double = {
      math.abs(avgValue - specificValue)
    }

    def deltaUdf = udf(deltaFunc _)


    val reviews_df6 = reviews_df5.withColumn("sentimentDelta" , deltaUdf(reviews_df5("avg(sentiment)"),reviews_df5("sentiment")))

    val reviews_df7 = reviews_df6.withColumn("overallDelta" , deltaUdf(reviews_df6("avg(star_rating)"),reviews_df6("star_rating")))


    // It was computing a/b in old datasets becuase helpful comumn was array .
    def computeHelpfulColumn(stringInt: Int) : Double = {
      stringInt*1.0
    }
    val computeHelpfulUdf = udf(computeHelpfulColumn _)

    val reviews_df8 = reviews_df7.withColumn("helpful_ratio", computeHelpfulUdf($"helpful_votes"))


    val assembler = new VectorAssembler()
      .setInputCols(Array("overallDelta", "sentimentDelta", "helpful_ratio"))
      .setOutputCol("features")

    val featuresDF = assembler.transform(reviews_df8)

    println("Feature combined using VectorAssembler")
    featuresDF.show()

    import org.apache.spark.ml.feature.MinMaxScaler

    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")

    val scalerModel = scaler.fit(featuresDF)

    val scaledData = scalerModel.transform(featuresDF)
    println(s"Features scaled to range: [${scaler.getMin}, ${scaler.getMax}]")

    scaledData.show()

    import org.apache.spark.ml.clustering.GaussianMixture


    var schema = types.StructType(
      StructField("cluster" , IntegerType, false) ::
        StructField("silhouette", DoubleType, false) :: Nil)

    var cluster_silhouette_df = sparkSession.createDataFrame(sc.emptyRDD[Row],schema)


    for( a <- 2 to 50){

      val gmm = new GaussianMixture()
        .setK(a).setFeaturesCol("scaledFeatures")

      val model = gmm.fit(scaledData)
      val predictions = model.transform(scaledData)

      val evaluator = new ClusteringEvaluator().setDistanceMeasure("cosine")
      val silhouette = evaluator.evaluate(predictions);
      println("silhouette value " + silhouette + " for clustering size " + a);

      val newRow = Seq((a ,silhouette)).toDF("cluster", "silhouette");
      cluster_silhouette_df = cluster_silhouette_df.union(newRow)


      for (i <- 0 until model.getK) {
        println(s"Gaussian $i:\nweight=${model.weights(i)}\n" +
          s"mu=${model.gaussians(i).mean}\nsigma=\n${model.gaussians(i).cov}\n")
      }


      val isNormal: Any => Boolean = _.asInstanceOf[DenseVector].toArray.exists(_ > 0.999)
      val isNormalUdf = udf(isNormal)
      val spammer_df = predictions.withColumn("normal", isNormalUdf($"probability"))

      spammer_df.show();

      val spammer_df2 = spammer_df.columns.foldLeft(spammer_df)((current, c) => current.withColumn(c, col(c).cast("String")))
      val spammer_df3 = spammer_df2.select("review_id","product_id", "customer_id", "prediction", "normal")

      spammer_df3.coalesce(1).write.mode(SaveMode.Overwrite).csv(outputFilePath + "_" + a);


      cluster_silhouette_df.show()
    }

    cluster_silhouette_df.coalesce(1).write.mode(SaveMode.Overwrite).csv(outputFilePath + "_silhouette_cluster");


  }

}
