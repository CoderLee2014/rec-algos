package robertlee.rec.ranking.utils

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import scopt.{DefaultOParserSetup, OParser, OParserSetup}

object SparkInit{
  def init(args: Array[String], appName: String): (SparkSession,Option[Config]) = {
      val setup: OParserSetup = new DefaultOParserSetup {
          override def showUsageOnError = Some(true)
      }
      val options = OParser.parse(ArgsParser.parser, args, Config(), setup)
      try{
          options.foreach(println)
      }catch {
        case e: Exception =>
          println("Print options exp.")
      }
      println(options.mkString("\n"))


      val conf = new SparkConf().set("spark.hadoop.validateOutputSpecs", "false")
      conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      //conf.registerKryoClasses(Array(classOf[XGBoostClassifier],classOf[XGBoostClassificationModel], classOf[Booster]))
      val sc = new SparkContext(conf)
    (SparkSession.builder().appName(appName).getOrCreate(), options)
    }

  def parseArgs(args: Array[String]): Option[Config] ={
    val setup: OParserSetup = new DefaultOParserSetup {
      override def showUsageOnError = Some(true)
    }
    val options = OParser.parse(ArgsParser.parser, args, Config(), setup)
    try{
      options.foreach(println)
    }catch {
      case e: Exception =>
        println("Print options exp.")
    }
    println(options.mkString("\n"))
    options
  }
}
