import java.util.Properties

import Sentiment.Sentiment
import edu.stanford.nlp.ling.CoreAnnotations
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations

import scala.collection.convert.wrapAll._

object SentimentAnalyzer {

  val properties = new Properties()
  properties.setProperty("annotators", "tokenize, ssplit, parse, sentiment")

  val pipeline = new StanfordCoreNLP(properties)

  def getSentiment(input: String): Sentiment = Option(input) match {
    case Some(text) if !text.isEmpty =>
      val (_, sentiment) = extractAllSentiment(text).maxBy { case (sentence, _) => sentence.length }
      return sentiment
    case _ => throw new IllegalArgumentException("Error: Input received is null or empty.")
  }

  def extractAllSentiment(text: String): List[(String, Sentiment)] = {
    val lines = pipeline.process(text).get(classOf[CoreAnnotations.SentencesAnnotation])
    return lines
      .map(line => (line, line.get(classOf[SentimentCoreAnnotations.SentimentAnnotatedTree])))
      .map { case (line, tree) => (line.toString, RNNCoreAnnotations.getPredictedClass(tree)) }
      .toList
  }

}