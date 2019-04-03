
package robertlee.rec.ranking.feature

import bee.simhash.main.Murmur3
import java.text.SimpleDateFormat
import java.util.concurrent.TimeUnit
import java.util.Date

import robertlee.rec.ranking.algos.XGBAlgo
import robertlee.rec.ranking.data.BasicData
import robertlee.rec.ranking.utils.{ArgsParser, Config}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import scopt.{DefaultOParserSetup, OParser, OParserSetup}

object FeatureInspector {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    //conf.registerKryoClasses(Array(classOf[MyClass1],classOf[MyClass2]))
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().appName("Ranking_xgb_lr").getOrCreate()

    val setup: OParserSetup = new DefaultOParserSetup {
      override def showUsageOnError = Some(true)
    }
    val options = OParser.parse(ArgsParser.parser, args, Config(),setup)
    println(options.mkString("\n"))

    println("start to load data from csv files: ")
    val raw_data = FeatureEngineering.loadRealTimeDataSet(options, spark, BasicData.SchemaRealtime,"train")


    println("Generating training dataset:")
    val data = FeatureEngineering.sampling(raw_data, 1)
      .drop("device_id")
      .drop("feed_id")
      .drop("label_date")
      .drop("timestamp")
      .na.fill("null:0.0")
      .na.fill(0.0)
    data.show(10)

    data.describe().show(false)
    val data_size = data.count()
    println(s"Training dataset size: ${data_size}")

    //hash_test(data)

    val xgb_algo = new XGBAlgo()
      xgb_algo.init(options, data.schema)
    val data_df = xgb_algo.transform(data, options)

    var features = xgb_algo._pipelineModel.transform(data_df)

    features.show(false)
     val ft =  features.rdd.map{
      x =>
        val vec = x.getAs[SparseVector]("features")
        Vectors.dense(vec.toDense.toArray)
    }
    features.describe().show(false)

    val summary: MultivariateStatisticalSummary = Statistics.colStats(ft)
    println(summary.min)  // a dense vector containing the mean value for each column
    println(summary.max)  // a dense vector containing the mean value for each column
    println(summary.mean)  // a dense vector containing the mean value for each column
    println(summary.variance)  // column-wise variance
    println(summary.numNonzeros)  // number of nonzeros in each column

    spark.stop()
  }

  def hash_test(data: DataFrame): Unit ={
        //count tags num
        val counter = data.select("category_tags_doc").na.drop().distinct().rdd.map{
          x =>
            (1, x(0).toString.split(","))
        }
        val res =  counter.reduceByKey((a,b)=> (a ++ b).distinct).map{
          x =>
            x._2
        }.collect()
        println(res)
        println(res(0).mkString(","))
        println("content_tags hashing:")
        var map = Map[Int, Int]()
        res(0).foreach{
          x =>
            println(x, FeatureEngineering.murmurHash(x, 12345) % 300)
            val hash_v = FeatureEngineering.murmurHash(x.substring(0,x.length-2), 12345) % 300
            if(map.contains(hash_v)){
              map = map + (hash_v -> (map(hash_v) + 1))
            }else{
              map = map + (hash_v -> 1)
            }
        }
        println("murmur32 bucket distribution:")
        map.foreach{
          x =>
            println(x._1,x._2)
        }
        var map_64 = Map[Int, Int]()
        println("murmursh 64 test:")
        res(0).foreach{
          x =>
            var hash_v64 = Murmur3.hash64(x.substring(0,x.length-2).getBytes()) % 150 + 150
            println(x, hash_v64)
            if(map_64.contains(hash_v64.toInt)){
              map_64 = map_64 + (hash_v64.toInt -> (map_64(hash_v64.toInt) + 1))
            }else{
              map_64 = map_64 + (hash_v64.toInt -> 1)
            }
        }
        println("hash 64 bucket distribution:")
        map_64.foreach{
          x =>
            println(x._1,x._2)
        }
  }

  def colMissValueCount(data: DataFrame): Unit ={
        val pos = data.where("label=1").cache()
        val neg = data.where("label=0").cache()
        val shrink_pos  = 1.0
        val c_pos = (pos.count()/shrink_pos).toInt
        val c_neg = neg.count()
        val ratio = if(c_pos*1.0*3/c_neg > 1.0) 1.0 else c_pos*1.0*3/c_neg
        val neg_train = neg.sample(false,fraction=ratio,seed=12345)
        val c_neg_train = neg_train.count
        for(col <- data.columns){
          val num_pos = pos.select(col).filter(pos(col).isNull || pos(col) === "" || pos(col).isNaN).count()
          val num_neg = neg_train.select(col).filter(neg_train(col).isNull || neg_train(col) === "" || neg_train(col).isNaN).count()
          println(s"${col} null number: ${num_pos*1.0 / c_pos}, ${num_neg*1.0 / c_neg_train}")
        }
  }
}
