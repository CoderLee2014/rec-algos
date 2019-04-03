package robertlee.rec.ranking.embedding

import java.text.SimpleDateFormat
import java.util.Date
import java.util.concurrent.TimeUnit

import robertlee.rec.ranking.data.BasicData
import robertlee.rec.ranking.feature.FeatureEngineering
import robertlee.rec.ranking.utils.{Config, SparkInit}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.collect_list

object UserEmbedding {
        def main(args: Array[String]): Unit ={
        val (spark, options) = SparkInit.init(args,"Embedding")
        generateUserDataSet(spark, options)
        }

        //Generate user2vec training data.
        def generateUserDataSet(spark:SparkSession, options:Option[Config]): Unit ={
import spark.implicits._
    val data = FeatureEngineering
            .generateDataSet(spark, options, BasicData.schema, "train")
            .where("label=1")
            .groupBy("feed_id")
            .agg(collect_list("device_id") as "users")
            .filter(row => row.getAs[Seq[String]]("users").size>1)
            .rdd.map(x=>x.getAs[Seq[String]]("users").mkString(" "))
            .toDF("users")
            data.repartition(1)
            .write.mode("overwrite")
            .option("sep","\t")
            .format("csv")
            .save(options.get.uservec_path + "/data_" + options.get.train_date.replaceAll("-","_") + ".txt")
            }

            def addUserEmbeddingFeature(spark: SparkSession, options: Option[Config], origin: DataFrame, name: String): DataFrame ={
            val sdf = new SimpleDateFormat("yyyy-MM-dd")
            val uservec = loadUserVectors(spark, options, name).collect().toMap
            val bc = spark.sparkContext.broadcast(uservec)
            val origin = FeatureEngineering.generateDataSet(spark, options, BasicData.schema, name)

            val rdd = origin.rdd.map{
            row =>
            val userid = row.getAs[String]("device_id")
            val dt = row.getAs[String]("label_date")
            val key = userid + "_" + sdf.format(new Date(sdf.parse(dt).getTime - TimeUnit.DAYS.toMillis(1)))
            val uservec = bc.value.getOrElse(key, new Array[Double](100))//Array[Double]
            Row.fromSeq(row.toSeq ++ Seq(uservec))
            }
            spark.createDataFrame(rdd, BasicData.schemaTrained_user2vec)
            }

            def loadUserVectors(spark: SparkSession, options: Option[Config], name:String): RDD[(String, Array[Double])] ={
            val sdf = new SimpleDateFormat("yyyy-MM-dd")
            var paths = List[String]()
            if(name.equals("train")){
            for( a <- 1 to (options.get.days+1)){
        var result = new Date(sdf.parse(options.get.train_date).getTime - TimeUnit.DAYS.toMillis(a))
        val date =  sdf.format(result)
        val path = "/data/robertlee/userVec_result/wordvec_" + date.replaceAll("-", "_") + ".txt"
        paths = paths :+ path
        }
        }else{
        for( a <- 0 to options.get.eval_days){
        var result = new Date(sdf.parse(options.get.eval_date).getTime + TimeUnit.DAYS.toMillis(a-1))
        val date =  sdf.format(result)
        val path = "/data/robertlee/userVec_result/wordvec_" + date.replaceAll("-", "_") + ".txt"
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
        val user_id = arr(1)
        val user_embedding = arr.takeRight(arr.size-2).map(_.toDouble)
        (user_id + "_"+ dt, user_embedding)
        }
        vectors
        }

        }
