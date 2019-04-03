package robertlee.rec.ranking.algos

import java.net.URI

import robertlee.rec.ranking.algos.LRAlgo.{maxIter}
import robertlee.rec.ranking.feature._
import robertlee.rec.ranking.utils.Config
import ml.combust.bundle.BundleFile
import ml.combust.bundle.serializer.SerializationFormat
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.bundle.SparkBundleContext
import ml.combust.mleap.spark.SparkSupport._
import org.apache.spark.ml.{Pipeline, PipelineModel, linalg}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.mleap.SparkUtil
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.max
import resource.managed
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.types.StructType

class LRAlgo extends Algo{
  var _pipelineModel: PipelineModel = _
  var _schema: StructType = _
  var _lrModel: LogisticRegressionModel = _

  def init(options: Option[Config], schema: StructType): Unit ={
    this._schema = schema
    if(!options.get.train_lr){
      loadModel(options)
    }
  }
  def fit(data: DataFrame, options: Option[Config]): PipelineModel = {
    val pipeline = buildPipeline()
    this._pipelineModel = pipeline.fit(data)
    this._lrModel = this._pipelineModel.stages(this._pipelineModel.stages.length-1).asInstanceOf[LogisticRegressionModel]
    savePipelineModel(options)
    saveBundleLRModel(options, this._pipelineModel.transform(data))
    this._pipelineModel
  }

  def transform(data: DataFrame, options: Option[Config]): DataFrame ={
    _pipelineModel.transform(data)
  }

  def evaluate(spark: SparkSession, options: Option[Config]): Unit = {
    //Load eval dataset
    val evalSet = FeatureEngineering.loadEvalDataset(spark, options, this._schema)
    LRAlgo.evaluate(this._pipelineModel, evalSet, options)
  }

  def evaluate(spark: SparkSession, options: Option[Config], xgb: XGBAlgo): Unit = {
    //Load eval dataset
    val evalSet = FeatureEngineering.loadEvalDataset(spark, options, this._schema)
    //Feature engineering.
    val input = xgb.transform(evalSet, options)
    println("After xgb model transformed: ")
    input.show(false)
    LRAlgo.evaluate(this._pipelineModel, input, options)
  }

  def savePipelineModel(options:Option[Config]): Unit ={
    val lr_pipeline_model_path = "viewfs://hadoop/data/robertlee/model_test/lr_pipeline_model_" + options.get.train_date + "_" + options.get.version
    this._pipelineModel.write.overwrite().save(lr_pipeline_model_path)
  }

  def saveBundleLRModel(options: Option[Config], data: DataFrame): Unit ={
     val sbc = SparkBundleContext().withDataset(data)
     val pipelineModel = SparkUtil.createPipelineModel(Array(this._pipelineModel))

    val tmp_file = "jar:file:/tmp/rankingxgblr" + options.get.train_date + "_" + options.get.version + ".bundle.zip"
    val tmp_f = "/tmp/rankingxgblr" + options.get.train_date + "_" + options.get.version + ".bundle.zip"
    for(bf <- managed(BundleFile(tmp_file))) {
      pipelineModel.writeBundle.format(SerializationFormat.Json).save(bf)(sbc).get
    }

    val modelSavePath_hdfs = "viewfs://hadoop/user/robertlee/ranking/xgb_lr/tmp_lr_pipeline_" + options.get.train_date + "_" + options.get.version + ".bundle.zip"
    val fs = FileSystem.get(new URI("viewfs://hadoop/data/robertlee/"),new Configuration())
    fs.copyFromLocalFile(new Path(tmp_f.toString), new Path(modelSavePath_hdfs))
    fs.close()
  }

  def loadModel(options: Option[Config]): Unit ={
    val model_path =  options.get.lr_pipeline_model_path
    this._pipelineModel = PipelineModel.load(model_path)
    this._lrModel = this._pipelineModel.stages(this._pipelineModel.stages.length-1).asInstanceOf[LogisticRegressionModel]

  }

  override def buildPipeline(): Pipeline = {
    val splits_feature1 = Array(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, Double.PositiveInfinity)
    val bucketizer_feature1 = new Bucketizer()
      .setInputCol("feature1")
      .setOutputCol("ctr_bucket_vec")
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
      .setOutputCol("feature4vec")
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

    val extract_feature14 = new ExtractKV()
      .setInputCol("feature14")
      .setOutputCol("feature14_key")
      .setIsExtractKey(true)
    val splits_feature14 = Array[Int](0,1,2,3).map(_.toString)
    val oneHot_feature14 = new OneHotEncoding()
      .setInputCol("feature14_key")
      .setOutputCol("feature14_vec")
      .setSplits(splits_feature14)

    val extract_feature15 = new ExtractKV()
      .setInputCol("feature15")
      .setOutputCol("feature15_key")
      .setIsExtractKey(true)
    val splits_feature15 = Array[Int](1,2,3).map(_.toString)
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
    val splits_feature17 = Array[Int](1,2,3,4,5,6,7,8).map(_.toString)
    val oneHot_feature17 = new OneHotEncoding()
      .setInputCol("feature17_key")
      .setOutputCol("feature17_vec")
      .setSplits(splits_feature17)

    val hasher_feature18 = new TagHashing()
      .setInputCol("feature18")
      .setOutputCol("feature18_vec")
      .setSeeds(12345)
      .setHashingDim(100)

    val hasher_feature19 = new TagHashing()
      .setInputCol("feature19")
      .setOutputCol("feature19_vec")
      .setSeeds(12345)
      .setHashingDim(1000)

    val hasher_feature20 = new TagHashing()
      .setInputCol("feature20")
      .setOutputCol("feature20_vec")
      .setSeeds(12345)
      .setHashingDim(100)

    val hasher_feature21 = new TagUnionHashing()
      .setInputCols(Array("short_feature21_tag_doc", "long_feature21_tag_doc"))
      .setOutputCol("feature21_vec")
      .setFilterPosTags(Array(true, false))
      .setTopKsTags(Array(100,50))
      .setHashingDim(50000)
      .setSeeds(54321)

    val hasher_feature22 = new TagUnionHashing()
      .setInputCols(Array("short_feature22_tag_doc", "long_feature22_tag_doc"))
      .setOutputCol("feature22_vec")
      .setFilterPosTags(Array(true, false))
      .setTopKsTags(Array(20,20))
      .setHashingDim(10000)
      .setSeeds(54321)

    val assembler_feature21 = new VectorAssembler()
      .setInputCols(Array("feature21"))
      .setOutputCol("feature21_vec")
    val assembler_feature22 = new VectorAssembler()
      .setInputCols(Array("feature22"))
      .setOutputCol("feature22_vec")
    val assembler_feature13 = new VectorAssembler()
      .setInputCols(Array("feature13"))
      .setOutputCol("feature13_vec")
    val assembler_feature14 = new VectorAssembler()
      .setInputCols(Array("feature14"))
      .setOutputCol("feature14_vec")

    val interaction_feature13 = new Interaction()
      .setInputCols(Array("feature21_vec", "feature13_vec"))
      .setOutputCol("interactedCol_feature13")

    val interaction_feature14 = new Interaction()
      .setInputCols(Array("feature22_vec", "feature14_vec"))
      .setOutputCol("interactedCol_feature14")

    val xgbleaf = new OneHotVecEncoding()
      .setInputCol("predLeaf")
      .setOutputCol("xgb_leaf")
      .setValuesSize(XGBAlgo.maxDepth)//XGBAlgo.maxDepth
      .setDim(XGBAlgo.num_trees)

    val assembler = new VectorAssembler()
      .setInputCols(
        LRAlgo.lr_input_features
      )
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setMaxIter(maxIter)
      .setRegParam(0.3)
      .setElasticNetParam(0.002)
      .setFeaturesCol("features")

    val stages = Array(
      //bucketizer_feature1,
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
      extract_feature14,
      extract_feature14,
      extract_feature16,
      extract_feature17,
      oneHot_feature17,
      hasher_feature18,
      hasher_feature19,
      hasher_feature20,
      hasher_feature21,
      hasher_feature22,
      assembler_feature21,
      xgbleaf,
      assembler,
      lr
    )
    val pipeline = new Pipeline()
      .setStages(stages)
    pipeline
  }
}
object LRAlgo {

  val maxIter = 50

  val lr_input_features =  Array(
    "feature1",
    "feature2"
    //....
  )

  def evaluate(model: PipelineModel, eval:DataFrame, options: Option[Config]): Unit ={
    val lrModel = model.stages(model.stages.length-1).asInstanceOf[LogisticRegressionModel]
    import eval.sparkSession.implicits._
    if(options.get.train_lr){
      val trainingSummary = lrModel.binarySummary

      // Obtain the objective per iteration.
      val objectiveHistory = trainingSummary.objectiveHistory
      println("objectiveHistory:")
      objectiveHistory.foreach(loss => println(loss))

      // Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
      val roc = trainingSummary.roc
      roc.show()
      println(s"areaUnderROC: ${trainingSummary.areaUnderROC}")

      // Set the model threshold to maximize F-Measure
      val fMeasure = trainingSummary.fMeasureByThreshold
      val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
      val bestThreshold = fMeasure.where($"F-Measure"=== maxFMeasure)
        .select("threshold").head().getDouble(0)
      lrModel.setThreshold(bestThreshold)
    }

    model.stages(model.stages.length-1) = lrModel
    val prediction = model.transform(eval)
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
