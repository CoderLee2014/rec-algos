package robertlee.rec.ranking.feature

import java.text.SimpleDateFormat
import java.util.Date
import java.util.concurrent.TimeUnit

import robertlee.rec.ranking.data.BasicData
import robertlee.rec.ranking.data.BasicData._
import robertlee.rec.ranking.utils.{Config, SparkInit}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.attribute.{Attribute, NumericAttribute}
import org.apache.spark.sql.types.StructType

object FeatureEngineering{

  val maxDepth = 8

  def main(args: Array[String]): Unit ={
      val (spark, options) = SparkInit.init(args,"feature_preprocessing")
  }


  def sampling(raw_data: DataFrame, rate: Int): DataFrame ={
    val pos = raw_data.where("label=1")
    val neg = raw_data.where("label=0")
    val shrink_pos  = 1.0
    val c_pos = (pos.count()/shrink_pos).toInt
    val c_neg = neg.count()
    val ratio = if(c_pos * 1.0 * rate/c_neg > 1.0) 1.0 else c_pos * 1.0 * rate / c_neg
    println(s"Postive samples: ${c_pos}")
    println(s"Negative samples: ${c_neg}")

    val data = pos.sample(false, 1/shrink_pos,seed=12345)
      .union(neg.sample(false,fraction=ratio,seed=12345))
    println("Data set size after sampled: " + data.count())
    data
  }

  def generateDataSet(spark: SparkSession, options: Option[Config], schema: StructType, name: String): DataFrame ={
    val sdf = new SimpleDateFormat("yyyy-MM-dd")
    var paths = List[String]()
    if(name.equals("train")){
      paths = paths :+ (options.get.data_path + "/dt=" + options.get.train_date + "/user_feature_face.txt")
      for( a <- 1 to options.get.days){
        var result = new Date(sdf.parse(options.get.train_date).getTime - TimeUnit.DAYS.toMillis(a))
        val date =  sdf.format(result)
        val path = options.get.data_path + "/dt=" + date + "/user_feature_face.txt"
        paths = paths :+ path
      }
      println("Generating training dataset:")
    } else{ //eval dataset.
      var date = options.get.eval_date
      for( a <- (0 to options.get.eval_days)){
        val result = new Date(sdf.parse(date).getTime + TimeUnit.DAYS.toMillis(1))
        date =  sdf.format(result)
        val path = options.get.data_path + "/dt=" + date + "/user_feature_face.txt"
        paths = paths :+ path
      }
      println("Generating eval dataset:")
    }

    println(paths)
    val raw_data = spark.read.format("csv").schema(schema)
      .option("delimiter", "\t").option("header", "false")
      .load(paths:_*)
    println("datasize loaded:",raw_data.count())

    val origin_data =  if(options.get.active_users)  filterLabelsV2(raw_data) else raw_data
    val data = FeatureEngineering.sampling(origin_data, 1)
      .na.fill("null:0.0")
      .na.fill(0.0)
    if(options.get.active_users){
      println("After getting active users: total samples:",data.count())
    }
    data
  }


  def loadEvalDataset(spark: SparkSession, options: Option[Config], schema: StructType): DataFrame = {
    if(options.get.eval_set.equals("hdfs")){
      generateDataSet(spark, options, schema, "eval")
    }
    else
      loadFromSeq(spark)
  }


  def loadFromSeq(spark: SparkSession): DataFrame ={
    import spark.implicits._
    spark.createDataFrame(Seq(
      //Record 1
     (
        "865176031238423",//"device_id",
        "119822427948",// "feed_id",
        "feature1",
        "feature2",
        0//"label"
      )
    )
    ).toDF(
      "device_id",
      "feed_id",
      "feature1",
      "feature2",
      "label"
    )
  }

  def loadRealTimeDataSet(options: Option[Config], spark: SparkSession, schema: StructType, name: String): DataFrame ={
    val sdf = new SimpleDateFormat("yyyy-MM-dd")
    //val result = new Date(sdf.parse(args(0)).getTime - TimeUnit.DAYS.toMillis(7))
    var paths = List[String]()
    if(name.equals("train")){
      for( a <- 0 to options.get.days){
        var result = new Date(sdf.parse(options.get.train_date).getTime - TimeUnit.DAYS.toMillis(a))
        val date =  sdf.format(result)
        val path = options.get.data_path + "/" + date + "/*/part-*"
        paths = paths :+ path
      }
    }else{
      for( a <- 0 to options.get.eval_days){
        var result = new Date(sdf.parse(options.get.eval_date).getTime + TimeUnit.DAYS.toMillis(a))
        val date =  sdf.format(result)
        val path = options.get.data_path + "/" + date + "/*/part-*"
        paths = paths :+ path
      }
    }
    println(paths)
    val raw_data = spark.read.format("csv").schema(schema)
      .option("delimiter", "\t").option("header", "false")
      .load(paths:_*)
    filterLabels(raw_data)
  }

  def filterLabels(data: DataFrame): DataFrame ={
    val cols = data.columns.toList
    data.select("label_date","device_id","feed_id","label")
      .dropDuplicates("label_date","device_id","feed_id","label")
      .groupBy("label_date","device_id","feed_id")
      .agg(max("label").alias("real_label"))
      .withColumnRenamed("label_date","label_date_1")
      .withColumnRenamed("device_id","device_id_1")
      .withColumnRenamed("feed_id","feed_id_1")
      .withColumnRenamed("real_label","label_1")
      .join(data.dropDuplicates("label_date","device_id","feed_id","label"),
          col("label_date_1")===col("label_date") &&
          col("device_id_1")===col("device_id") &&
          col("feed_id_1")===col("feed_id") &&
          col("label_1")===col("label"),
        "inner")
      .select(cols(0),cols.takeRight(cols.size-1):_*)
  }

  //filtering users with only one record.
  def filterLabelsV2(data: DataFrame): DataFrame ={
    //way 1 :data.where("activeuser='1'")
    val users = data.groupBy("device_id").agg({"feed_id"->"count"}).where("count(feed_id)>1").select("device_id")
    //val feeds = data.groupBy("feed_id").agg({"device_id"})
    users.join(data,"device_id")
  }

  import org.apache.spark.sql.functions._
  def explode(df: DataFrame, features: String): (DataFrame, Array[String]) ={
    val separator = ","
    lazy val first = df.select(features).first()
    val numAttrs = first.toString().split(separator).length
    val attributes: Array[Attribute] = {
      Array.tabulate(numAttrs)(i => NumericAttribute.defaultAttr.withName("col" + "_" + i))
    }
    val attrs = Array.tabulate(numAttrs)(n => "col_" + n)
    //    println(attrs.mkString(","))
    val fieldCols = attributes.zipWithIndex.map(x => {
      val assembleFunc = udf {
        str: String =>
          //println(str.split(separator).mkString(","))
          str.split(separator)(x._2)
      }
      assembleFunc(df(features)).as(x._1.name.get, x._1.toMetadata())
    })
    val newDF = df.select(col("*") +: fieldCols: _*)
    newDF.show()
    (newDF, attrs)
  }

  def getValue(v :String): Double ={
    try {
      v.split(":")(0).toDouble
    }
    catch {
      case e: Exception => -1
    }
  }


  def PosTagFiltering(x: String): Boolean ={
    var cond = false;
    try {
      if (x.split(":")(1).toDouble >= 0.0)
        cond = true
    }
    catch {
      case e: Exception =>
        if(x.split(":")(0).equals("null") || x.split(":")(0).equals(""))
          cond = false
        else
          cond = true // tags without scores.
    };
    cond
  }

  def NullFiltering(x: String): (String,Double) ={
    var city = "null"
    var prob = 0.0
    try {
      if (x.split(":")(1).toDouble>=0)
        city = x.split(":")(0)
      (city, prob)
    }
    catch {
      case e: Exception => (city, prob)
    }
  }

  def murmurHash(word: String, seed: Int): Int = {
    val c1 = 0xcc9e2d51
    val c2 = 0x1b873593
    val r1 = 15
    val r2 = 13
    val m = 5
    val n = 0xe6546b64

    var hash = seed //12345

    for (ch <- word.toCharArray) {
      var k = ch.toInt
      k = k * c1
      k = (k << r1) | (hash >> (32 - r1))
      k = k * c2

      hash = hash ^ k
      hash = (hash << r2) | (hash >> (32 - r2))
      hash = hash * m + n
    }

    hash = hash ^ word.length
    hash = hash ^ (hash >> 16)
    hash = hash * 0x85ebca6b
    hash = hash ^ (hash >> 13)
    hash = hash * 0xc2b2ae35
    hash = hash ^ (hash >> 16)

    hash
  }
  import scala.util.control.Breaks._

  def hashing(words: Array[String],bucket_size:Int, seed: Int): DenseVector = {
    var bk = new Array[Double](bucket_size)
    try{
      breakable {
        for (word <- words) {
          val term = word.split(":")(0)
          if (term.equals("null")||term.equals("")) {
            break
          }
          var weight = "1.0"
          if (word.split(":").size == 2) {
            weight = word.split(":")(1)
          }
          var bucket = murmurHash(term, seed) % bucket_size
          try{
            if (weight.toDouble > 0)
              bk(bucket) = 1.0
          }catch{
            case _ : NumberFormatException => bk(bucket) = 1.0
          }
        }
      }
    }catch{
      case e : Exception =>
        bk = bk.map(_ => 0.0)
    }

    new DenseVector(bk)
  }

  def stringHashing(word: String,bucket_size:Int, seed: Int): DenseVector = {
    var bk = new Array[Double](bucket_size)
    try{
          val bucket = murmurHash(word, seed) % bucket_size
          bk(bucket) = 1.0
    }catch{
      case e : Exception =>
        bk = bk.map(_ => 0.0)
    }
    new DenseVector(bk)
  }

  def hashingWeighted(words: Array[String],bucket_size:Int, seed: Int): DenseVector = {
    var buckets = new Array[Double](bucket_size)
    try{
      breakable {
        for (word <- words) {
          val term = word.split(":")(0)
          if (term.equals("null")|| term.equals("")) {
            break
          }
          val bucket = murmurHash(term, seed) % bucket_size
          buckets(bucket) = 1.0
        }
      }
    }catch{
      case e : Exception =>
        println("hw exp: ", e)
        println("hw words: ", words)
        buckets = buckets.map(_ => 0.0)
    }
    new DenseVector(buckets)
  }

  def getPosTags(row: Row, field_name: String, top: Int): Array[String] ={
    try {
      row.getAs[String](field_name).split(",").filter(PosTagFiltering).take(top).map(x => x.split(":")(0))
    }catch{
      case e: Exception =>
        println(s"${field_name} processing exp: ", row.getAs[String](field_name));
        Array("null")
    }
  }

  def getTags(row: Row, field_name: String): Array[String] ={
    try {
      row.getAs[String](field_name).split(",").map(x => x.split(":")(0))
    }catch{
      case e: Exception =>
        println(s"${field_name} processing exp: ", row.getAs[String](field_name));
        Array("null")
    }
  }

  def getActiveUsers(df: DataFrame): DataFrame ={
    df.groupBy("device_id").agg(countDistinct("label_date") as "active_days").where("active_days>=3").select("device_id")
  }

  def oneHotEncoding(split: Array[Double], value: Double): Array[Double] ={
    var i = -1
    var arr = new Array[Double](split.size-1).map(_=>0.0)
    split.foreach{
      threshold =>
        if(value >= threshold)
          i += 1
    }
    if(i>=0 && i<arr.size)
      arr(i) = 1
    arr
  }
}
