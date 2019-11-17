import java.util.Properties

import edu.stanford.nlp.ling.CoreAnnotations
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{Row, SaveMode, SparkSession, types}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.collection.convert.wrapAll._


object ProjectHandler {
  def main(args: Array[String]): Unit = {

    Logger.getRootLogger.setLevel(Level.WARN)

    if (args.length < 2) {
      println("Usage:  InputFilePath OutputFilePath")
      System.exit(1)
    }

    val inputFilePath = args(0)
    val outputFilePath = args(1)


    // create Spark context with Spark configuration
    //    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("ProjectMain");   //Local
    val sparkConf = new SparkConf().setAppName("FakeReviewsClassification");                       //AWS

    val sc = new SparkContext(sparkConf)

    val sparkSession = SparkSession.builder
      .config(conf = sparkConf)
      .appName("FakeReviewsClassification")
      .getOrCreate()

    sc.setLogLevel("ERROR")

    import org.apache.spark.sql.functions._
    import sparkSession.implicits._
    import scala.util.matching.Regex
    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.linalg.DenseVector
    import org.apache.spark.ml.evaluation._


    val raw_reviews_df = sparkSession.read.csv(inputFilePath)

    val generateCompositeId = udf( (first: String, second: String, third: String) => { first + "_" + second + "_" + third } )

    //generating ids for each review
    val reviews_df1  = raw_reviews_df.withColumn("review_id", generateCompositeId($"product_id", $"customer_id", $"review_date"))

    reviews_df1.cache();

    // Extracting Sentiment value for each review
    val reviews_text_df = reviews_df1.select("review_id", "review_body")
    def analyzeSentiment: (String => Int) = { s => this.mainSentiment(s) }
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
    val product_avg_sentiment_score_df1 = product_avg_sentiment_score_df.groupBy('asin).agg(asinSentimentMap);
    product_avg_sentiment_score_df1.show()
    product_avg_sentiment_score_df1.cache()

    val product_avg_sentiment_score_df2 = product_avg_sentiment_score_df1.drop("avg(product_id)")


    println("product_avg_sentiment_score_df2 completed");

    //calculating average overall review score for the product
    val product_avg_overall_df = reviews_df3.select("product_id", "star_rating")
    val asinOverallMap = product_avg_overall_df.columns.map((_ -> "mean")).toMap
    val product_avg_overall_df1 = product_avg_overall_df.groupBy('asin).agg(asinOverallMap);
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



    def computeHelpfulColumn(wrappedaArr: mutable.WrappedArray[BigDecimal]): Double = {
      val test = wrappedaArr.toString();

      val numPattern = new Regex("(\\d+)")
      val matches = numPattern.findAllIn(test).toArray.map(_.toDouble);

      if( matches(2) == 0 ){
        return 0.0
      }else{
        matches(0)/matches(2)
      }
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


  } //end of main

  //NLP Properties
  val props = new Properties()
  props.setProperty("annotators", "tokenize, ssplit, parse, sentiment")
  val pipeline: StanfordCoreNLP = new StanfordCoreNLP(props)

  def mainSentiment(input: String): Int = Option(input) match {
    case Some(text) if !text.isEmpty => {
      var sentiment:Int  = extractSentiment(text)
      return sentiment }
    case _ => throw new IllegalArgumentException("input can't be null or empty")
  }

  private def extractSentiment(text: String): Int = {
    val (_, sentiment) = extractSentiments(text)
      .maxBy { case (sentence, _) => sentence.length }
    sentiment
  }

  def extractSentiments(text: String): List[(String, Int)] = {
    val annotation: Annotation = pipeline.process(text)
    val sentences = annotation.get(classOf[CoreAnnotations.SentencesAnnotation])
    sentences
      .map(sentence => (sentence, sentence.get(classOf[SentimentCoreAnnotations.SentimentAnnotatedTree])))
      .map { case (sentence, tree) => (sentence.toString, RNNCoreAnnotations.getPredictedClass(tree)) }
      .toList
  }

}
