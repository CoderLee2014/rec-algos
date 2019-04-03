package robertlee.rec.ranking.algos

import robertlee.rec.ranking.data.BasicData
import robertlee.rec.ranking.feature.FeatureEngineering
import robertlee.rec.ranking.utils.{SparkInit}
import org.apache.spark.sql.{DataFrame}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics


object XGBLR {

  def main(args: Array[String]): Unit = {

    val (spark, options) = SparkInit.init(args, "Ranking_xgb_lr")

    val dataSchema = BasicData.schema

    println("start to load data from csv files: ")
    //val raw_data = FeatureEngineering.loadDataFromHive(options, spark, BasicData.SchemaRealtime)
    import spark.implicits._
    val data =  if(options.get.train_xgb||options.get.train_lr)
                    FeatureEngineering.generateDataSet(spark, options, dataSchema, "train")
                else  List(1,2,3,4,5).toDF


    val xgb_algo = new XGBAlgo()
    xgb_algo.init(options, dataSchema)
    if(options.get.train_xgb)
      xgb_algo.fit(data, options)
    if(options.get.eval_xgb)
      xgb_algo.evaluate(spark, options)

    //Generate xgb leaf features.

    val lr_algo = new LRAlgo()
    lr_algo.init(options, dataSchema)
    if(options.get.train_lr) {
      val lr_input = xgb_algo.transform(data, options)
      println("print data_lr_input")
      lr_input.show(false)

      lr_algo.fit(lr_input, options)
    }
    lr_algo.evaluate(spark, options, xgb_algo)

    spark.stop()
  }

  def evaluate(model: PipelineModel, prediction: DataFrame): Unit ={
    try{
      // Batch prediction
      prediction.show(false)
      prediction.select("features",
        "label",
        "rawPrediction",
        "probability",
        "prediction").show(false)

      val predictionAndLabels = prediction.select("probability","label")
        .rdd.map{
        x
        => (x.getAs[org.apache.spark.ml.linalg.DenseVector](0).values(1), x(1).toString.toDouble)
      }

      // Instantiate metrics object
      val metrics = new BinaryClassificationMetrics(predictionAndLabels,10)

      // Precision by threshold
      val precision = metrics.precisionByThreshold
      precision.foreach { case (t, p) =>
        println(s"Threshold: $t, Precision: $p")
      }

      // Recall by threshold
      val recall = metrics.recallByThreshold
      recall.foreach { case (t, r) =>
        println(s"Threshold: $t, Recall: $r")
      }

      // Precision-Recall Curve
      val PRC = metrics.pr

      // F-measure
      val f1Score = metrics.fMeasureByThreshold
      f1Score.foreach { case (t, f) =>
        println(s"Threshold: $t, F-score: $f, Beta = 1")
      }

      val beta = 0.5
      val fScore = metrics.fMeasureByThreshold(beta)
      fScore.foreach { case (t, f) =>
        println(s"Threshold: $t, F-score: $f, Beta = 0.5")
      }

      // AUPRC
      val auPRC = metrics.areaUnderPR
      println(s"Area under precision-recall curve = $auPRC")

      // Compute thresholds used in ROC and PR curves
      val thresholds = precision.map(_._1)

      // ROC Curve
      val roc = metrics.roc

      // AUROC
      val auROC = metrics.areaUnderROC
      println(s"Area under ROC = $auROC")

    }catch {
      case e: Exception => println("Metrics exception:", e)
    }


    val evaluator = new MulticlassClassificationEvaluator()
    //evaluator.setLabelCol("classIndex")
    evaluator.setPredictionCol("prediction")
    val accuracy = evaluator.evaluate(prediction)
    println("The model accuracy is : " + accuracy)
  }

}
