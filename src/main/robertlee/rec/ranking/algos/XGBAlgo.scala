package robertlee.rec.ranking.algos

import java.net.URI

import robertlee.rec.ranking.algos.XGBAlgo._
import robertlee.rec.ranking.data.BasicData
import robertlee.rec.ranking.feature._
import robertlee.rec.ranking.utils.{Config, SparkInit}
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{Bucketizer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

class XGBAlgo extends Algo{

  var _xgb_model_input_features: Array[String] = _
  var _pipelineModel: PipelineModel = _
  var _schema: StructType = _
  var _xgbModel: XGBoostClassificationModel = _



  def init(options:Option[Config], schema: StructType): Unit ={
    this._schema = schema
    this._xgb_model_input_features = XGBAlgo.xgb_model_input_features
    if(!options.get.train_xgb)
      loadModel(options)
  }

  def fit(data: DataFrame, options:Option[Config]): PipelineModel = {
    val pipeline = buildPipeline()
    this._pipelineModel = pipeline.fit(data)
    savePipelineModel(options)
    this._xgbModel = this._pipelineModel.stages(this._pipelineModel.stages.length-1).asInstanceOf[XGBoostClassificationModel]
    //println(this._xgbModel.nativeBooster.getModelDump().mkString("\n"))
    saveNativeXGBModel(options)
    this._pipelineModel
  }
  def transform(data: DataFrame, options: Option[Config]): DataFrame ={
    val booster = this._pipelineModel.stages(this._pipelineModel.stages.length-1).asInstanceOf[XGBoostClassificationModel].nativeBooster
    val pipeline = new Pipeline().setStages(this._pipelineModel.stages.take(this._pipelineModel.stages.length-1))
    val output_1 = pipeline.fit(data.limit(1000)).transform(data).cache()
    println("After xgb pipeline, input raw show:")
    output_1.show(false)
    val xgb_output  = XGBUtils.transformLeaf(booster,output_1, data.sparkSession)
    println("xgb output show:")
    xgb_output.show(false)
    xgb_output
  }

  def evaluate(spark: SparkSession, options: Option[Config]): Unit = {
    //Load eval dataset
    val evalSet = FeatureEngineering.loadEvalDataset(spark, options, this._schema)
    XGBAlgo.evaluate(this._pipelineModel, evalSet)
  }

  def savePipelineModel(options:Option[Config]): Unit ={
    val model_path =  options.get.model_path + "/xgb_pipeline_model_" + options.get.train_date + "_" + options.get.version
    this._pipelineModel.write.overwrite().save(model_path)
  }

  def saveNativeXGBModel(options: Option[Config]): Unit ={
    val fs = FileSystem.get(new URI(options.get.active_node +"/data/robertlee/"),new Configuration())
    val nativeModelPath = fs.create(new Path(options.get.model_path + "/xgb_model_"  + options.get.train_date + "_" + options.get.version))
    this._xgbModel.nativeBooster.saveModel(nativeModelPath)
    fs.close()
  }

  def loadModel(options: Option[Config]): Unit ={
    val model_path =  options.get.xgb_pipeline_model_path
    this._pipelineModel = PipelineModel.load(model_path)
    this._xgbModel = this._pipelineModel.stages(this._pipelineModel.stages.length-1).asInstanceOf[XGBoostClassificationModel]
  }

  def buildPipeline(): Pipeline ={

    val splits_feature1 = Array(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, Double.PositiveInfinity)
    val bucketizer_feature1 = new Bucketizer()
      .setInputCol("feature1")
      .setOutputCol("feature1_bucket_vec")
      .setSplits(splits_feature1)

    val splits_feature2 = Array[Int](1,9).map(_.toString)
    val oneHot_feature2 = new OneHotEncoding()
      .setInputCol("feature2")
      .setOutputCol("feature2_vec")
      .setSplits(splits_feature2)

    val splits_feature3 = Array[Int](0,3,5,6,7,8).map(_.toString)
    val oneHot_feature3 = new OneHotEncoding()
      .setInputCol("feature3")
      .setOutputCol("feature3_vec")
      .setSplits(splits_feature3)

    val splits_feature4 = Array[Int](1,2,3,4,5,6).map(_.toString)
    val oneHot_feature4 = new OneHotEncoding()
      .setInputCol("feature4")
      .setOutputCol("feature4_vec")
      .setSplits(splits_feature4)

    val splits_feature5 = Array[Int](1,2).map(_.toString)
    val oneHot_feature5 = new OneHotEncoding()
      .setInputCol("feature5")
      .setOutputCol("feature5_vec")
      .setSplits(splits_feature5)

    val splits_feature6 = Array[Int](0,2,3).map(_.toString)
    val oneHot_feature6 = new OneHotEncoding()
      .setInputCol("feature6")
      .setOutputCol("feature6_vec")
      .setSplits(splits_feature6)


    val splits_feature7 = Array(0.0, 5, 10, 20, 30,Double.PositiveInfinity)
    val bucketizer_feature7 = new Bucketizer()
      .setInputCol("feature7")
      .setOutputCol("feature7_bucketed")
      .setSplits(splits_feature7)

    val splits_feature8 = Array(Double.NegativeInfinity,100, 1000, 10000, 100000, 1000000, Double.PositiveInfinity)
    val bucketizer_feature8 = new Bucketizer()
      .setInputCol("feature8")
      .setOutputCol("feature8_bucketed")
      .setSplits(splits_feature8)

    val splits_feature9 = Array(Double.NegativeInfinity,100, 1000, 5000, 10000,Double.PositiveInfinity)
    val bucketizer_feature9 = new Bucketizer()
      .setInputCol("feature9")
      .setOutputCol("feature9_bucketed")
      .setSplits(splits_feature9)

    val splits_feature10 = Array(Double.NegativeInfinity,100, 1000, 5000, 10000,Double.PositiveInfinity)
    val bucketizer_feature10 = new Bucketizer()
      .setInputCol("feature10")
      .setOutputCol("feature10_bucketed")
      .setSplits(splits_feature10)

    val splits_feature11 = Array(Double.NegativeInfinity,100, 1000, 5000, 10000, Double.PositiveInfinity)
    val bucketizer_feature11 = new Bucketizer()
      .setInputCol("feature11")
      .setOutputCol("feature11_bucketed")
      .setSplits(splits_feature11)

    val hasher_feature12 = new TagHashing()
      .setInputCol("feature12")
      .setOutputCol("feature12_vec")
      .setHashingDim(1000)
      .setSeeds(12345)

    val hasher_feature13 = new TagHashing()
      .setInputCol("feature13_doc")
      .setOutputCol("feature13_vec")
      .setHashingDim(50000)
      .setSeeds(54321)

    val hasher_feature14 = new TagHashing()
      .setInputCol("feature14_doc")
      .setOutputCol("feature14_vec")
      .setHashingDim(10000)
      .setSeeds(12345)

    val extract_feature15 = new ExtractKV()
      .setInputCol("feature15")
      .setOutputCol("feature15_key")
      .setIsExtractKey(true)
    val splits_feature15 = Array[Int](0,1,2,3).map(_.toString)
    val oneHot_feature15 = new OneHotEncoding()
      .setInputCol("feature15_key")
      .setOutputCol("feature15_vec")
      .setSplits(splits_feature15)

    val extract_feature16 = new ExtractKV()
      .setInputCol("feature16")
      .setOutputCol("feature16_key")
      .setIsExtractKey(true)
    val splits_feature16 = Array[Int](1,2,3).map(_.toString)
    val oneHot_feature16 = new OneHotEncoding()
      .setInputCol("feature16_key")
      .setOutputCol("feature16_vec")
      .setSplits(splits_feature16)

    val extract_feature17 = new ExtractKV()
      .setInputCol("feature17")
      .setOutputCol("feature17_key")
      .setIsExtractKey(true)
    val splits_feature17 = Array[Int](1,2,3).map(_.toString)
    val oneHot_feature17 = new OneHotEncoding()
      .setInputCol("feature17_key")
      .setOutputCol("feature17_vec")
      .setSplits(splits_feature17)

    val extract_feature18 = new ExtractKV()
      .setInputCol("feature18")
      .setOutputCol("feature18_key")
      .setIsExtractKey(true)
    val splits_feature18 = Array[Int](1,2,3,4,5,6,7,8).map(_.toString)
    val oneHot_feature18 = new OneHotEncoding()
      .setInputCol("feature18_key")
      .setOutputCol("feature18_vec")
      .setSplits(splits_feature18)

    val hasher_feature19 = new TagHashing()
      .setInputCol("feature19")
      .setOutputCol("feature19_vec")
      .setSeeds(12345)
      .setHashingDim(100)

    val hasher_feature20 = new TagHashing()
      .setInputCol("feature20")
      .setOutputCol("feature20_vec")
      .setSeeds(12345)
      .setHashingDim(1000)

    val hasher_feature21 = new TagHashing()
      .setInputCol("feature21")
      .setOutputCol("feature21_vec")
      .setSeeds(12345)
      .setHashingDim(100)

    val hasher_feature22 = new TagUnionHashing()
      .setInputCols(Array("short_feature22_tag_doc", "long_feature22_tag_doc"))
      .setOutputCol("feature22_vec")
      .setFilterPosTags(Array(true, false))
      .setTopKsTags(Array(100,50))
      .setHashingDim(50000)
      .setSeeds(54321)

    val hasher_feature23 = new TagUnionHashing()
      .setInputCols(Array("short_feature23_tag_doc", "long_feature23_tag_doc"))
      .setOutputCol("feature23_vec")
      .setFilterPosTags(Array(true, false))
      .setTopKsTags(Array(20,20))
      .setHashingDim(10000)
      .setSeeds(54321)

    val assembler = new VectorAssembler()
      .setInputCols(
        xgb_model_input_features
      )
      .setOutputCol("features")

    val booster = new XGBoostClassifier(
      Map("eta" -> 0.1f,
        "max_depth" -> XGBAlgo.maxDepth,
        "objective" -> "binary:logistic",
        "num_round" -> XGBAlgo.num_trees,
        "num_workers" -> 30,
        "lambda" -> 10,
        "min_child_weight" -> 3,
        "gamma" -> 1,
        "subsample" -> 0.8,
        "colsample_bytree" -> 0.8
        //        ,
        //        "checkpoint_path" -> checkpointPath,
        //        "checkpoint_interval" -> 1
      )
    )
    booster.setFeaturesCol("features")

    val stages = Array(
      bucketizer_feature1,
      oneHot_feature2,
      oneHot_feature3,
      oneHot_feature4,
      oneHot_feature5,
      oneHot_feature6,
      bucketizer_feature7,
      bucketizer_feature8,
      bucketizer_feature9,
      bucketizer_feature10,
      bucketizer_feature11,
      hasher_feature12,
      hasher_feature13,
      hasher_feature14,
      extract_feature15,
      extract_feature16,
      extract_feature17,
      extract_feature18,
      oneHot_feature15,
      oneHot_feature16,
      oneHot_feature17,
      oneHot_feature18,
      hasher_feature19,
      hasher_feature20,
      hasher_feature21,
      hasher_feature22,
      hasher_feature23,
      assembler,
      booster
    )
    val pipeline = new Pipeline()
      .setStages(stages)
    pipeline
  }
}

object XGBAlgo {

  val maxDepth = 6
  val num_trees = 300

  val xgb_model_input_features = Array(
    "feature1",
    "feature2"
    //...
  )

  def main(args: Array[String]): Unit = {

    val (spark, options) = SparkInit.init(args, "Ranking_GBTFM")
    val dataSchema = BasicData.schema
    import spark.implicits._
    val data =  if(options.get.train_xgb||options.get.train_fm){
      val origin = FeatureEngineering.generateDataSet(spark, options, dataSchema, "train")
      //Feed embedding
      //FeedEmbedding.addFeedEmbeddingFeature(spark, options, origin, "train")
      origin
    }
    else  List(1,2,3,4,5).toDF

    //val Array(train, test) = data.randomSplit(Array(0.8, 0.2))

    val xgb_algo = new XGBAlgo()
    xgb_algo.init(options, dataSchema)
    if(options.get.train_xgb)
      xgb_algo.fit(data, options)
    if(options.get.eval_xgb)
      xgb_algo.evaluate(spark, options)


    println("save model .")
    println("Feature Importances .")
    println(xgb_algo._xgbModel.nativeBooster.getFeatureScore())

    spark.stop()
  }



  def evaluate(model: PipelineModel, test: DataFrame): Unit ={
    try{
      // Batch prediction
      val prediction = model.transform(test)
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
    val accuracy = evaluator.evaluate(model.transform(test))
    println("The model accuracy is : " + accuracy)
  }

}
