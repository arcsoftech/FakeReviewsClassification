

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
    val sparkSession = SparkSession.builder
      .config(conf = sparkConf)
      .appName("FakeReviewsClassification")
      .getOrCreate()
    import sparkSession.implicits._

    sc.setLogLevel("ERROR")

    if (args.length < 2) {
      println("Usage:  Input Output")
      System.exit(1)
    }

    val input = args(0)
    val output = args(1)
    var printFlag = true
    if (args(3) == 0) {
      printFlag = false
    }


val parquetFileDF = sparkSession.read.parquet(input)

// Parquet files can also be used to create a temporary view and then used in SQL statements
parquetFileDF.createOrReplaceTempView("parquetFile")
var query = "SELECT * FROM parquetFile LIMIT "+ args(2)
val original_df = sparkSession.sql(query)

    // val original_df = sparkSession.read.option("inferSchema", "true").option("header", "true").csv(input)

    val uniqueIdGenerator = udf((product_id: String, customer_id: String, review_date: String) => {
      product_id + "_" + customer_id + "_" + review_date
    })

    val computed_df1 = original_df.withColumn("review_id", uniqueIdGenerator($"product_id", $"customer_id", $"review_date"))
    computed_df1.cache()

    // computing sentiment review

    val reviews_text_df = computed_df1.select("review_id", "review_body")

    def sentimentAnalysis: (String => Int) = { s => SentimentAnalyzer.mainSentiment(s) }

    val sentimentAnalysisUDF = udf(sentimentAnalysis)

    //Dropping text review column after extracting sentiment
    val computed_df2 = computed_df1.select("review_id", "product_id", "helpful_votes", "star_rating", "customer_id", "review_date", "review_body");

    computed_df2.cache()
    if (printFlag) {
      computed_df2.show()
    }


    // Gnerating sentiment column.

    val computed_df3 = computed_df2.withColumn("sentiment", sentimentAnalysisUDF(reviews_text_df("review_body"))).cache.drop("review_body")
    computed_df3.cache()
    if (printFlag) {
      computed_df3.show()
    }


    // Find average sentiment score  and average rating for each product

    val average_sentiment_rating_score_df = computed_df3.select("product_id", "sentiment", "star_rating")
    val productIdSentimentMap = average_sentiment_rating_score_df.columns.map((_ -> "mean")).toMap
    val average_sentiment_rating_score_df1 = average_sentiment_rating_score_df.groupBy("product_id").agg(productIdSentimentMap);
    average_sentiment_rating_score_df1.cache()
    if (printFlag) {
      average_sentiment_rating_score_df1.show()
    }


    val average_sentiment_rating_score_df2 = average_sentiment_rating_score_df1.drop("avg(product_id)")


    val computed_df5 = computed_df3.join(average_sentiment_rating_score_df2, Seq("product_id"))

    // Function to compute the distance of each datapoint from its mean.S
    def meanDistance(avgValue: Double, specificValue: Double): Double = {
      math.abs(avgValue - specificValue)
    }

    def meanDistanceUDF = udf(meanDistance _)

    val computed_df6 = computed_df5.withColumn("sentimentDelta", meanDistanceUDF(computed_df5("avg(sentiment)"), computed_df5("sentiment")))
    val computed_df7 = computed_df6.withColumn("overallDelta", meanDistanceUDF(computed_df6("avg(star_rating)"), computed_df6("star_rating")))


    // Generate feature vector
    val assembler = new VectorAssembler()
      .setInputCols(Array("overallDelta", "sentimentDelta", "helpful_votes"))
      .setOutputCol("features")

    val featuresDF = assembler.transform(computed_df7)
    if (printFlag) {
      println("Feature combined using VectorAssembler")
      featuresDF.show()
    }


    // Min Max Standardization
    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("standardizedfeatures")

    val scalerModel = scaler.fit(featuresDF)

    val scaledData = scalerModel.transform(featuresDF)

    if (printFlag) {
      println(s"Features scaled to range: [${scaler.getMin}, ${scaler.getMax}]")
      scaledData.show()
    }


    // custom csvSchema defined for csv
    var csvSchema = types.StructType(
      StructField("cluster", IntegerType, false) ::
        StructField("silhouette_score", DoubleType, false) :: Nil)

    var clusterSilhouette_df = sparkSession.createDataFrame(sc.emptyRDD[Row], csvSchema)


    // Compute silhouette_score value against K clusters ranging from 2 to 50
    for (a <- 2 to 50) {

      val gausian_mixture_model = new GaussianMixture()
        .setK(a).setFeaturesCol("standardizedfeatures").setMaxIter(100)

      val model = gausian_mixture_model.fit(scaledData)
      val predictions = model.transform(scaledData)

      val evaluator = new ClusteringEvaluator().setDistanceMeasure("cosine")
      val silhouette_score = evaluator.evaluate(predictions);
      println("silhouette_score value " + silhouette_score + " for K " + a);

      val newRow = Seq((a, silhouette_score)).toDF("cluster", "silhouette_score");
      clusterSilhouette_df = clusterSilhouette_df.union(newRow)


      for (i <- 0 until model.getK) {
        println(s"Gaussian $i:\nweight=${model.weights(i)}\n" +
          s"mu=${model.gaussians(i).mean}\nsigma=\n${model.gaussians(i).cov}\n")
      }


      val isNormal: Any => Boolean = _.asInstanceOf[DenseVector].toArray.exists(_ > 0.90)
      val isNormalUdf = udf(isNormal)
      val reviewer_fake_df = predictions.withColumn("normal", isNormalUdf($"probability"))

      if (printFlag) {
        reviewer_fake_df.show();
      }


      val reviewer_fake_df2 = reviewer_fake_df.columns.foldLeft(reviewer_fake_df)((current, c) => current.withColumn(c, col(c).cast("String")))
      val reviewer_fake_df3 = reviewer_fake_df2.select("review_id", "product_id", "customer_id", "prediction", "normal")

      reviewer_fake_df3.coalesce(1).write.mode(SaveMode.Overwrite).csv(output + "_" + a);

      if (printFlag) {
        clusterSilhouette_df.show()
      }

    }

    clusterSilhouette_df.coalesce(1).write.mode(SaveMode.Overwrite).csv(output + "_silhouette_scoreVScluster");
  }

}
