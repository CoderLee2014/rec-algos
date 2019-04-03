package robertlee.rec.ranking.embedding

import java.text.SimpleDateFormat
import java.util.Date
import java.util.concurrent.TimeUnit

import robertlee.rec.ranking.data.BasicData
import robertlee.rec.ranking.feature.FeatureEngineering
import org.apache.spark.sql.functions._
import robertlee.rec.ranking.utils.{Config, SparkInit}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import robertlee.rec.ranking.feature.CosineSimilarity

class FeedEmbedding {

}

object FeedEmbedding{
  def main(args: Array[String]): Unit ={
    val (spark, options) = SparkInit.init(args,"Embedding")
    generateFeedDataSet(spark, options)
  }


  def generateFeedDataSet(spark:SparkSession, options:Option[Config]): Unit ={
    import spark.implicits._
    val data = FeatureEngineering
      .generateDataSet(spark, options, BasicData.schema, "train")
      .where("label=1")
      .groupBy("device_id").agg(collect_list("feed_id") as "feeds")
      .filter(row => row.getAs[Seq[String]]("feeds").size>1)
      .rdd.map(x=>x.getAs[Seq[String]]("feeds").mkString(" "))
      .toDF("feeds")
    data.repartition(1)
      .write.mode("overwrite")
      .option("sep","\t")
      .format("csv")
      .save(options.get.itemvec_path + "/data_" + options.get.train_date.replaceAll("-","_") + ".txt")
  }

  def generateUserHistList(spark:SparkSession, options:Option[Config]): Unit ={
    import spark.implicits._
    FeatureEngineering.generateDataSet(spark, options, BasicData.schema, "train")
      .where("label=1")
      .groupBy("device_id").agg(collect_list("feed_id") as "feeds")
      .rdd.map(x=>(x.getAs[String]("device_id"),x.getAs[Seq[String]]("feeds").mkString(" "), options.get.train_date))
      .toDF("device_id","feeds", "dt")
      .write.mode("overwrite")
      .option("sep","\t")
      .format("csv")
      .save("/data/robertlee/userHistList/dt=" + options.get.train_date + "/data.txt")
  }

  def addFeedEmbeddingFeature(spark: SparkSession, options: Option[Config], origin: DataFrame, name: String): DataFrame ={
    val sdf = new SimpleDateFormat("yyyy-MM-dd")
    val feedvec = loadFeedVectors(spark, options, name).collect().toMap
    val bc = spark.sparkContext.broadcast(feedvec)

    val rdd = origin.rdd.map{
      row =>
        val feedid = row.getAs[String]("feed_id")
        val dt = row.getAs[String]("label_date")
        val key = feedid + "_" + sdf.format(new Date(sdf.parse(dt).getTime - TimeUnit.DAYS.toMillis(1)))
        val feedvec = bc.value.getOrElse(key, new Array[Double](100))//Array[Double]
        Row.fromSeq(row.toSeq ++ Seq(feedvec))
    }
    spark.createDataFrame(rdd, BasicData.schemaTrained_feed2vec)
  }



  def addUserHistListSim(spark: SparkSession, options: Option[Config], origin: DataFrame, name: String): DataFrame = {
    import spark.implicits._
    val sdf = new SimpleDateFormat("yyyy-MM-dd")
    val feedvec = FeedEmbedding.loadFeedVectors(spark, options, name).collect().toMap
    val bc = spark.sparkContext.broadcast(feedvec)
    val userHistList = FeedEmbedding.loadUserHistList(spark, options, name)
      .rdd.map{
      row =>
        val deviceid = row.getAs[String]("device_id")
        val feeds = row.getAs[String]("feeds")
        val dt = row.getAs[String]("dt")
        val dt_new = sdf.format(new Date(sdf.parse(dt).getTime - TimeUnit.DAYS.toMillis(1)))
        (deviceid, feeds, dt_new)
    }.toDF("deviceid", "feeds", "dt")
    val rdd = origin.join(userHistList,origin("device_id")===userHistList("deviceid")&& origin("label_date")===userHistList("dt"), "left")
      .drop("dt")
      .drop("deviceid")
      .rdd.map{
          row =>
          val feedid = row.getAs[String]("feed_id")
          val deviceid = row.getAs[String]("device_id")
          val dt = row.getAs[String]("label_date")
          val dt_subOne = "_" + sdf.format(new Date(sdf.parse(dt).getTime - TimeUnit.DAYS.toMillis(1)))
          val key_vec = feedid + dt_subOne
          val feedvec = bc.value.getOrElse(key_vec, new Array[Double](100))//Array[Double]
          val userHist =
          try{
                row.getAs[String]("feeds").split(" ")
                  .map {
                        x =>
                        if (x.equals("")) 0.0
                        else {
                          val feedvec_i = bc.value.getOrElse(x + dt_subOne, new Array[Double](100))
                          CosineSimilarity.cosineSimilarity(feedvec_i, feedvec)
                        }
                  }
          }catch {case e: Exception => Array(0.0)}
          val sim = userHist.sum / userHist.length
          Row.fromSeq(row.toSeq.take(row.toSeq.size-1) ++ Seq(feedvec) ++ Seq(sim))
    }
    spark.createDataFrame(rdd, BasicData.schemaTrained_feed2vec)
  }

  def loadFeedVectors(spark: SparkSession, options: Option[Config], name: String): RDD[(String, Array[Double])] ={
    val sdf = new SimpleDateFormat("yyyy-MM-dd")
    var paths = List[String]()
    if(name.equals("train")){
      for( a <- 1 to (options.get.days+1)){
        var result = new Date(sdf.parse(options.get.train_date).getTime - TimeUnit.DAYS.toMillis(a))
        val date =  sdf.format(result)
        val path ="/data/robertlee/itemVec_result/wordvec_" + date.replaceAll("-", "_") + ".txt"
        paths = paths :+ path
      }
    }else{
      for( a <- 0 to options.get.eval_days){
        var result = new Date(sdf.parse(options.get.eval_date).getTime + TimeUnit.DAYS.toMillis(a-1))
        val date =  sdf.format(result)
        val path ="/data/robertlee/itemVec_result/wordvec_" + date.replaceAll("-", "_") + ".txt"
        paths = paths :+ path
      }
    }
    println(paths)
    val vectors = spark
      .sparkContext
      .textFile(paths.mkString(","))
      .map {
        row =>
          val arr = row.split(" ")
          val dt = arr(0).replaceAll("_","-")
          val feed_id = arr(1)
          val feed_embedding = arr.takeRight(arr.size-2).map(_.toDouble)
          (feed_id + "_"+ dt, feed_embedding)
      }
    vectors
  }



  def loadUserHistList(spark: SparkSession, options: Option[Config], name: String): DataFrame ={
    val sdf = new SimpleDateFormat("yyyy-MM-dd")
    var paths = List[String]()
    if(name.equals("train")){
      for( a <- 1 to (options.get.days+1)){
        var result = new Date(sdf.parse(options.get.train_date).getTime - TimeUnit.DAYS.toMillis(a))
        val date =  sdf.format(result)
        val path ="/data/robertlee/userHistList/dt=" + date + "/data.txt"
        paths = paths :+ path
      }
    }else{
      for( a <- 0 to options.get.eval_days){
        var result = new Date(sdf.parse(options.get.eval_date).getTime + TimeUnit.DAYS.toMillis(a-1))
        val date =  sdf.format(result)
        val path ="/data/robertlee/userHistList/dt=" + date + "/data.txt"
        paths = paths :+ path
      }
    }
    println(paths)
    val raw_data = spark.read.format("csv").schema(new StructType(Array(
      StructField("device_id", StringType, true),
      StructField("feeds", StringType, true),
      StructField("dt", StringType, true))))
      .option("delimiter", "\t").option("header", "false")
      .load(paths:_*)
    raw_data.cache()
    println(" UserHistList datasize loaded:",raw_data.count())
    raw_data
  }

}
