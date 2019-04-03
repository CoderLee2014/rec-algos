package robertlee.rec.ranking.algos

import robertlee.rec.ranking.data.BasicData
import robertlee.rec.ranking.embedding.{FeedEmbedding, UserEmbedding}
import robertlee.rec.ranking.utils.{Config, SparkInit}
import robertlee.rec.ranking.feature.FeatureEngineering

object GBTFM{

  val xgb_model_input_features = Array(
    "feature1",
    "feature2"
    //....
  )

  def main(args: Array[String]): Unit = {

    val (spark, options) = SparkInit.init(args, "Ranking_GBTFM")
    val rawDataSchema = BasicData.schema


    println("start to load data from csv files: ")
    import spark.implicits._
    val data =  if(options.get.train_xgb||options.get.train_fm){
      val origin = FeatureEngineering.generateDataSet(spark, options, rawDataSchema, "train")
      //Feed embedding
      FeedEmbedding.addFeedEmbeddingFeature(spark, options, origin, "train")

      //User embedding
      //UserEmbedding.addUserEmbeddingFeature(spark, options, origin, "train")
    }
    else  List(1,2,3,4,5).toDF

    println("After adding feed_vec: ")
    data.show(5,false)

    val xgb_algo = new XGBAlgo()
    xgb_algo.init(options, rawDataSchema)
    if(options.get.train_xgb)
      xgb_algo.fit(data, options)
    if(options.get.eval_xgb)
      xgb_algo.evaluate(spark, options)


    val fm_algo =  new FMAlgo()
    fm_algo.init(options, rawDataSchema)
    if(options.get.train_fm) {
      val fm_input = xgb_algo.transform(data, options).cache()
      println("print data_fm_input: " + fm_input.count())
      fm_input.show(false)
      fm_algo.fit(fm_input, options)
      fm_algo._model.save(spark.sparkContext, options.get.model_path + "_" + options.get.train_date + "_" + options.get.version)
      fm_algo._model.saveNative(options)
      printParams(fm_algo)
    }
    //Original model evaluation
    fm_algo._model.saveNative(options)
    fm_algo.evaluate(spark, options, xgb_algo)
    println("finished original model evaluation.")

    spark.stop()
  }
  def printParams(algo: FMAlgo): Unit ={
    println("FM model params:")
    println("k0, true")
    println("k1, true")
    println("intercept", algo._model.intercept)
    println("numFactors", algo._model.numFactors)
    println("numFeatures", algo._model.numFeatures)
    println("task", algo._model.task)
    println("max", algo._model.max)
    println("min", algo._model.min)
    println("weightVector", algo._model.weightVector.toArray.take(10).map(_.toArray.take(1)).mkString(","))
    println("m_sum includes num of zeros: ", algo._model.numFactors)
    println("m_sum_sqrt includes num of zeros: ", algo._model.numFactors)
    println("factorMatrix: ", algo._model.factorMatrix.toArray.take(10).mkString(","))
  }
}
