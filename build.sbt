name := "FakeReviewsClassification"

version := "0.1"

scalaVersion := "2.11.8"

val sparkVersion = "2.4.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-hive" % sparkVersion,
  "org.scala-lang" % "scala-library" % "2.11.12",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.9.2" artifacts (Artifact("stanford-corenlp", "models"), Artifact("stanford-corenlp"))
)
assemblyMergeStrategy in assembly := {
  case PathList("META-INF" , xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}