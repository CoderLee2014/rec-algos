package robertlee.rec.ranking.algos

import robertlee.rec.ranking.algos.FMAlgo._
import robertlee.rec.ranking.data.BasicData
import robertlee.rec.ranking.embedding.FeedEmbedding
import robertlee.rec.ranking.feature._
import robertlee.rec.ranking.utils.{Config, SparkInit}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, _}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{DenseVector}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{FMModel, FMWithLBFGS, FMWithSGD, LabeledPoint}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}


class FMAlgo extends Algo{
  var _model:FMModel = _
  var _schema: StructType = _
  var _pipelineModel:PipelineModel = _
  var _fp_output_cols: Array[String] = _
  var _fm_input_features: Array[String] = _


  def init(options:Option[Config], schema: StructType): Unit ={
    this._schema = schema
    this._fm_input_features = FMAlgo.vectorAsCols_GBT
    if(!options.get.train_fm){
      this._model = FMModel.loadNative(options)
    }
  }

  def fit(data: DataFrame, options: Option[Config]): PipelineModel = {
    _pipelineModel = buildPipeline().fit(data)

    val input = _pipelineModel.transform(data)

    val formatSamples = input.rdd.map( row =>{
      new LabeledPoint(row.getAs[Int]("label").toDouble, Vectors.fromML(row.getAs[DenseVector]("scaledFeatures")))
    }).cache()
    val regs = options.get.regParamFM.split(",").map(_.toDouble)
    _model = if(options.get.optimizer.equals("SGD")) FMWithSGD.train(formatSamples, task = options.get.task, numIterations = options.get.numIterFM, stepSize = options.get.stepSizeFM, miniBatchFraction = options.get.MiniBatchFraction, dim = (true, true, options.get.dim), regParam = (regs(0),regs(1),regs(2)), initStd = 0.1)
    else FMWithLBFGS.train(formatSamples, task = options.get.task, numIterations = options.get.numIterFM, numCorrections=options.get.numCorrections, dim = (true, true, options.get.dim), regParam = (regs(0),regs(1),regs(2)), initStd = 0.1)
    this._pipelineModel
  }

  def transform(samples:DataFrame, options:Option[Config]):DataFrame = {
    val preparedSamples = _pipelineModel.transform(samples)
    _model.predict(preparedSamples)
  }

  //Evaluate single model.
  def evaluate(spark: SparkSession, options: Option[Config], data: DataFrame): Unit = {
    println("Evaluate single FM model.")
    FMAlgo.evaluate(this._pipelineModel, this._model, data)
  }

  //Evaluate gbt_FM model.
  def evaluate(spark: SparkSession, options: Option[Config], xgb: XGBAlgo): Unit = {
    val origin = FeatureEngineering.generateDataSet(spark, options, this._schema, "eval").cache()
    val evalSet = FeedEmbedding.addUserHistListSim(spark, options, origin, "eval")

    //Feature engineering.
    val input = xgb.transform(evalSet, options)
    var predict_input: DataFrame = null
    try{
      predict_input = this._pipelineModel.transform(input)
    }catch {
      case e: Exception =>
        println("Build pipeline model.")
        val vectorAssembler = new VectorAssembler().setInputCols(FMAlgo.vectorAsCols_GBT).setOutputCol("scaledFeatures")
        predict_input = vectorAssembler.transform(input)
    }
    val predictions = this._model.predict(predict_input)
    FMAlgo.evaluate(predictions)
    println("Pos Metrics:")
    FMAlgo.evaluate(predictions.where("label=1"))
    println("Neg Metrics:")
    FMAlgo.evaluate(predictions.where("label=0"))
    evalSet.unpersist()
  }

  override def loadModel(options: Option[Config]): Unit = {

  }

  override def evaluate(spark: SparkSession, options: Option[Config]): Unit = ???

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


    val vector_cols =  vectorAsCols
    val vectorAssembler = new VectorAssembler()
      .setInputCols(vector_cols)
      .setOutputCol("scaledFeatures")

    val pipelineStages = Array(
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
      vectorAssembler
    )
    new Pipeline().setStages(pipelineStages)
  }
}

object FMAlgo{

  val vectorAsCols = Array(
    "feature2_vec",
    "feature3_vec",
    "feature24_vec",
    "feature13_vec",
    "feature14_vec",
    //"feature25_vec",
    //"feature26",
    "feature7_bucketed_vec",
    "feature8_bucketed_vec","feature9_bucketed_vec","feature10_bucketed_vec", "feature11_bucketed_vec",
    //"feature1_1","feature1_2",
    "feature1_bucket_vec",
    //"feature1_14",
    //"feature1_30",
    //          "feature8_1","feature8_2","feature8_7","feature8_14","feature8_30",
    //          "feature9_1","feature9_2","feature9_7","feature9_14","feature9_30",
    //          "feature10_1","feature10_2","feature10_7","feature10_14","feature10_30",
    //          "feature11_1","feature11_2","feature11_7","feature11_14","feature11_30",
    "feature4_vec",
    "feature5_vec",
    "feature6_vec",
    "feature15_vec",
    "feature16_vec",
    "feature17_vec",
    "feature18_vec",
    "feature19_vec",
    "feature20_vec",
    "feature21_vec",
    "feature12_vec",
    "feature22_vec",
    "feature23_vec"
  )

  val vectorAsCols_GBT = Array(
    "feature2_vec",
    "feature3_vec",
    "feature24_vec",
    "feature13_vec",
    "feature14_vec",
    //"feature25_vec",
    //"feature26",
    "feature7_bucketed_vec",
    "feature8_bucketed_vec","feature9_bucketed_vec","feature10_bucketed_vec", "feature11_bucketed_vec",
    //"feature1_1","feature1_2",
    "feature1_bucket_vec",
    //"feature1_14",
    //"feature1_30",
    //          "feature8_1","feature8_2","feature8_7","feature8_14","feature8_30",
    //          "feature9_1","feature9_2","feature9_7","feature9_14","feature9_30",
    //          "feature10_1","feature10_2","feature10_7","feature10_14","feature10_30",
    //          "feature11_1","feature11_2","feature11_7","feature11_14","feature11_30",
    "feature4_vec",
    "feature5_vec",
    "feature6_vec",
    "feature15_vec",
    "feature16_vec",
    "feature17_vec",
    "feature18_vec",
    "feature19_vec",
    "feature20_vec",
    "feature21_vec",
    "feature12_vec",
    "feature22_vec",
    "feature23_vec",
    "xgbleaf"//,
    //"sim"
  )

  def main(args: Array[String]): Unit = {

    val (spark, options) = SparkInit.init(args, "Ranking_FM")
    val dataSchema = BasicData.schema

    println("Generating training dataset:")
    import spark.implicits._
    val data =  if(options.get.train_xgb||options.get.train_fm)
      FeatureEngineering.generateDataSet(spark, options, dataSchema, "train").cache()
    else  List(1,2,3,4,5).toDF

    println("datasize loaded:", data.count())
    data.show(5)
    println("data date: " + options.get.train_date)

    if(options.get.tuning)
      tuning(data, options).save(spark.sparkContext, options.get.model_path + "_" + options.get.train_date + "_" + options.get.version)
    else{
      val Array(train, test) = data.randomSplit(Array(0.8, 0.2));
      val fm = new FMAlgo()
      fm.init(options, dataSchema)
      fm.fit(train, options)
      evaluate(fm._pipelineModel, fm._model, test)
      fm._model.saveNative(options)
      println("Model saved successfully!")
    }
    spark.stop()
  }

  def tuning(data: DataFrame, options:Option[Config]): FMModel ={

    var params:Map[String,Any] = Map(
      "--numIterFM" -> Array(100,200,300),
      "--dim" -> Array(2,4,6,8),
      //"--stepSizeFM" -> Array(0.01, 0.15, 0.2, 0.3),
      "--regParamFM" -> Array(0.1, 0.2, 0.3,0.5,0.8,1.0,1.2),
      "--numCorrections" -> Array(4,5,6,7,8,9),
      "--optimizer" -> Array("SGD")
    )

    val args :Array[Array[String]] =
      for(iter <- params("--numIterFM").asInstanceOf[Array[Int]];
          dim <- params("--dim").asInstanceOf[Array[Int]];
          //step <- params("--stepSizeFM").asInstanceOf[Array[Double]];
          reg <- params("--regParamFM").asInstanceOf[Array[Double]];
          numCor <- params("--numCorrections").asInstanceOf[Array[Int]];
          opt <- params("--optimizer").asInstanceOf[Array[String]]
      )
        yield Array(
          "--numIterFM", iter.toString,
          "--dim", dim.toString,
          //"--stepSizeFM", step.toString,
          "--regParamFM", "0,0,"+reg.toString,
          "--numCorrections", numCor.toString,
          "--optimizer", opt.toString,
          "--tuning","true"
        )
    var num = 0
    val models: Array[(FMModel, Map[String, Double], String)] =
      for(arg <- args)
        yield  {
          num +=  1
          val Array(train, test) = data.randomSplit(Array(0.8, 0.2))
          val fm = new FMAlgo()
          println(s"Training model $num  with params: \n" + arg.mkString("\n"))
          fm.fit(train, SparkInit.parseArgs(arg))
          val metrics = evaluate(fm._pipelineModel, fm._model, test)
          println("Metrics:")
          metrics.foreach(println)
          (fm._model, metrics, arg.mkString("\n"))
        }
    val bestRes = models.sortBy(_._2("acc")).reverse(0)
    println("Best model params: \n" + bestRes._3)
    val path = new Path(options.get.model_path +"/fmtuning_bestModelParams.txt")
    val conf = new Configuration()
    val fs = FileSystem.get(conf)
    val os = fs.create(path, true)
    os.write(bestRes._3.getBytes)
    os.write(bestRes._2.mkString("\n").getBytes)
    fs.close()
    bestRes._1
  }

  def calcuteInnerProduct(row: Row, a:DenseVector, b:DenseVector): Double ={
    val userEmbedding = a
    val itemEmbedding = b
    var aSquare = 0.0
    var bSquare = 0.0
    var abProduct = 0.0

    for (i <-0 until userEmbedding.size){
      aSquare += userEmbedding(i) * userEmbedding(i)
      bSquare += itemEmbedding(i) * itemEmbedding(i)
      abProduct += userEmbedding(i) * itemEmbedding(i)
    }
    var innerProduct = 0.0
    if (aSquare == 0 || bSquare == 0){
      innerProduct = 0.0
    }else{
      innerProduct = abProduct / (Math.sqrt(aSquare) * Math.sqrt(bSquare))
    }
    innerProduct
  }





  def evaluate(prediction: DataFrame): Map[String,Double] ={
    import prediction.sparkSession.implicits._
    var results = Map[String,Double]()
    try{
      // Batch prediction
      prediction.show(false)
      prediction.select(
        "label",
        "probability").show(false)

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
      results += ("auc" -> auROC)

      val evaluator = new MulticlassClassificationEvaluator()
      //evaluator.setLabelCol("classIndex")
      evaluator.setPredictionCol("probability")
      val accuracy = evaluator.evaluate(predictionAndLabels.map(x => if( x._1 > 0.5) (1.0,x._2) else (0.0,x._2)).toDF("probability","label"))
      println("The model accuracy is : " + accuracy)
      results += ("acc" -> accuracy)
      results
    }catch {
      case e: Exception => println("Metrics exception:", e)
        results
    }

  }


  def evaluate(pipelineModel: PipelineModel, fm: FMModel, data: DataFrame): Map[String, Double] ={
    import data.sparkSession.implicits._
    var results = Map[String,Double]()
    val test = pipelineModel.transform(data)
    val prediction = fm.predict(test)
    prediction.show(false)
    prediction.select(
      "label",
      "probability"
    ).show(false)
    val predictionAndLabels = prediction.select("probability","label")
      .rdd.map{
      x
      => (x.getAs[org.apache.spark.ml.linalg.DenseVector](0).values(1), x(1).toString.toDouble)
    }

    try{
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
      results += ("auPRC" -> auPRC)

      // Compute thresholds used in ROC and PR curves
      val thresholds = precision.map(_._1)

      // ROC Curve
      val roc = metrics.roc

      // AUROC
      val auROC = metrics.areaUnderROC
      println(s"Area under ROC = $auROC")
      results += ("auc" -> auROC)
    }catch {
      case e: Exception => println("Metrics exception:", e)
    }

    val evaluator = new MulticlassClassificationEvaluator()
    //evaluator.setLabelCol("classIndex")
    evaluator.setPredictionCol("probability")
    val accuracy = evaluator.evaluate(predictionAndLabels.map(x => if( x._1 > 0.5) (1.0,x._2) else (0.0,x._2)).toDF("probability","label"))
    println("The model accuracy is : " + accuracy)
    results += ("acc" -> accuracy)
    results
  }
}



