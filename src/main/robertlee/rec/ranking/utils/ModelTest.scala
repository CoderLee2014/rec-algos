package robertlee.rec.ranking.utils

import robertlee.rec.ranking.algos.{FMAlgo, GBTFM, LRAlgo, XGBAlgo}
import robertlee.rec.ranking.data.BasicData

object ModelTest{
  def main(args: Array[String]): Unit = {

    val (spark, options) = SparkInit.init(args, "Ranking_xgb_lr")
    val dataSchema = BasicData.schema

    val xgb_algo = new XGBAlgo()
    xgb_algo.init(options, dataSchema)
    if(options.get.eval_xgb)
      xgb_algo.evaluate(spark, options)

    val fm_algo =  new FMAlgo()
    fm_algo.init(options, dataSchema)
    GBTFM.printParams(fm_algo)
    fm_algo.evaluate(spark, options, xgb_algo)

    val lr_algo = new LRAlgo()
    lr_algo.init(options, dataSchema)
    lr_algo.evaluate(spark, options, xgb_algo)

    spark.stop()
  }
}
