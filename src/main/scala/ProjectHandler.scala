

// Scala Project to compute fake vs legitimate review using gaussian mixture model.
// Group Member:
// Arihant
// Anish
// Pawan
// Shobhit

import SentimentAnalyzer.extractSentiment
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, ColumnName, Row, SaveMode, SparkSession, types}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.clustering.GaussianMixture
import org.apache.spark.ml.evaluation._



object ProjectHandler {
  def main(args: Array[String]): Unit = {
    Logger.getRootLogger.setLevel(Level.WARN)
    val sparkConf = new SparkConf().setAppName("FakeReviewsClassification").set("spark.sql.broadcastTimeout", "36000"); //AWS
    val sc = new SparkContext(sparkConf)
    val Spark = SparkSession.builder
      .config(conf = sparkConf)
      .appName("FakeReviewsClassification")
      .getOrCreate()

    import Spark.implicits._
    sc.setLogLevel("ERROR")

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

    val computedDataFrame1 = original_df.withColumn("review_id", uniqueIdGenerator($"product_id", $"customer_id", $"review_date"))
    computedDataFrame1.cache()

    // computing sentiment review

    val reviews_text_df = computedDataFrame1.select("review_id", "review_body")

    def sentimentAnalysis: (String => Int) = { s => SentimentAnalyzer.mainSentiment(s) }

    val sentimentAnalysisUDF = udf(sentimentAnalysis)

    //Dropping text review column after extracting sentiment
    val computedDataFrame2 = computedDataFrame1.select("review_id", "product_id", "helpful_votes", "star_rating", "customer_id", "review_date", "review_body");

    computedDataFrame2.cache()
    if (printFlag) {
      computedDataFrame2.show()
    }


    // Gnerating sentiment column.

    val computedDataFrame3 = computedDataFrame2.withColumn("sentiment", sentimentAnalysisUDF(reviews_text_df("review_body"))).cache.drop("review_body")
    computedDataFrame3.cache()
    if (printFlag) {
      computedDataFrame3.show()
    }


    // Find average sentiment score  and average rating for each product

    val average_sentiment_rating_score_df = computedDataFrame3.select("product_id", "sentiment", "star_rating")
    val productIdSentimentMap = average_sentiment_rating_score_df.columns.map((_ -> "mean")).toMap
    val average_sentiment_rating_score_df1 = average_sentiment_rating_score_df.groupBy("product_id").agg(productIdSentimentMap);
    average_sentiment_rating_score_df1.cache()
    if (printFlag) {
      average_sentiment_rating_score_df1.show()
    }


    val average_sentiment_rating_score_df2 = average_sentiment_rating_score_df1.drop("avg(product_id)")


    val computedDataFrame5 = computedDataFrame3.join(average_sentiment_rating_score_df2, Seq("product_id"))

    // Function to compute the distance of each datapoint from its mean.S
    def meanDistance(mu: Double, data: Double): Double = {
      math.abs(mu - data)
    }

    def meanDistanceUDF = udf(meanDistance _)

    val computedDataFrame6 = computedDataFrame5.withColumn("sentimentDelta", meanDistanceUDF(computedDataFrame5("avg(sentiment)"), computedDataFrame5("sentiment")))
    val computedDataFrame7 = computedDataFrame6.withColumn("overallDelta", meanDistanceUDF(computedDataFrame6("avg(star_rating)"), computedDataFrame6("star_rating")))


    // Generate feature vector
    val assembler = new VectorAssembler()
      .setInputCols(Array("overallDelta", "sentimentDelta", "helpful_votes"))
      .setOutputCol("features")

    val featuresDF = assembler.transform(computedDataFrame7)
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


    // Compute silhouette width value against K clusters ranging from 2 to 50
    for (k <- 2 to 50) {

      val gausian_mixture_model = new GaussianMixture()
        .setK(k).setFeaturesCol("standardizedfeatures").setMaxIter(100)

      val model = gausian_mixture_model.fit(transformedData)
      val estimated_value = model.transform(transformedData)

      val model_evaluator = new ClusteringEvaluator().setDistanceMeasure("cosine")
      val s_width = model_evaluator.evaluate(estimated_value);
      println("silhouette width " + s_width + " for K " + k);

      val newLine = Seq((k, s_width)).toDF("cluster", "s_width");
      clusterSilhouette_df = clusterSilhouette_df.union(newLine)


      for (i <- 0 until model.getK) {
        println(s"Gaussian $i:\nweight=${model.weights(i)}\n" +
          s"mu=${model.gaussians(i).mean}\nsigma=\n${model.gaussians(i).cov}\n")
      }


      val checkNormalDistributionConfidence: Any => Boolean = _.asInstanceOf[DenseVector].toArray.exists(_ > 0.90)
      val checkNormalDistributionConfidenceUdf = udf(checkNormalDistributionConfidence)
      val reviewerDataFrame = estimated_value.withColumn("normal", checkNormalDistributionConfidenceUdf($"probability"))

      if (printFlag) {
        reviewerDataFrame.show();
      }


      val reviewerDataFrame2 = reviewerDataFrame.columns.foldLeft(reviewerDataFrame)((current, c) => current.withColumn(c, col(c).cast("String")))
      val reviewerDataFrame3 = reviewerDataFrame2.select("review_id", "product_id", "customer_id", "prediction", "normal")

      reviewerDataFrame3.coalesce(1).write.mode(SaveMode.Overwrite).csv(output + "_" + k);

      if (printFlag) {
        clusterSilhouette_df.show()
      }

    }

    clusterSilhouette_df.coalesce(1).write.mode(SaveMode.Overwrite).csv(output + "_silhouette_scoreVScluster");
  }

}
