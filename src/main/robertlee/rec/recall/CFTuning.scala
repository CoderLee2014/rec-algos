package robertlee.rec.recall

import java.text.SimpleDateFormat
import java.util.Date
import java.util.concurrent.TimeUnit

import robertlee.rec.ranking.utils.{Config, SparkInit}
import org.apache.spark.ml.{Estimator, Pipeline}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RankingEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.recommendation.{ALS, ALSExt, ALSModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.{SparkConf, SparkContext}

object CFTuning{

  val SchemaRealtime = new StructType(
    Array(
      StructField("device_id", StringType, true),
      StructField("feed_id", StringType, true),
      StructField("avg_kd_duration_1", DoubleType, true),
      //StructField("avg_kd_duration_2", DoubleType, true),
      StructField("avg_kd_duration_7", DoubleType, true),
      //StructField("avg_kd_duration_14", DoubleType, true),StructField("avg_kd_duration_30", DoubleType, true),
      StructField("short_interest_content_tag_doc", StringType, true),
      StructField("short_interest_category_tag_doc", StringType, true),
      StructField("long_interest_content_tag_doc", StringType, true),
      StructField("long_interest_category_tag_doc", StringType, true),
      StructField("feed_type", IntegerType, true),
      StructField("feed_pic_num", IntegerType, true),
      StructField("content_tags_doc", StringType, true),
      StructField("category_tags_doc", StringType, true),
      StructField("has_title", BooleanType, true),
      StructField("title_size", IntegerType, true),
      StructField("pub_days", IntegerType, true),
      StructField("read_count", IntegerType, true), StructField("like_count", IntegerType, true),StructField("comment_count", IntegerType, true),StructField("share_count", IntegerType, true),
      //StructField("ctr_1", DoubleType, true),StructField("ctr_2", DoubleType, true),
      StructField("ctr_7", DoubleType, true),//StructField("ctr_14", DoubleType, true),StructField("ctr_30", DoubleType, true),
      //      StructField("read_count_1", IntegerType, true),StructField("read_count_2", IntegerType, true),StructField("read_count_7", IntegerType, true),StructField("read_count_14", IntegerType, true),StructField("read_count_30", IntegerType, true),
      //      StructField("like_count_1", IntegerType, true),StructField("like_count_2", IntegerType, true),StructField("like_count_7", IntegerType, true),StructField("like_count_14", IntegerType, true),StructField("like_count_30", IntegerType, true),
      //      StructField("comment_count_1", IntegerType, true),StructField("comment_count_2", IntegerType, true),StructField("comment_count_7", IntegerType, true),StructField("comment_count_14", IntegerType, true),StructField("comment_count_30", IntegerType, true),
      //      StructField("share_count_1", IntegerType, true),StructField("share_count_2", IntegerType, true),StructField("share_count_7", IntegerType, true),StructField("share_count_14", IntegerType, true),StructField("share_count_30", IntegerType, true),
      StructField("label", IntegerType, true),
      StructField("label_date", StringType, true),
      StructField("timestamp", StringType, true),
      StructField("r_source", StringType, true),
      StructField("age", IntegerType, true),
      StructField("sex", IntegerType, true),
      StructField("recom_level", IntegerType, true),
      //face features
      StructField("income_of_baidu", StringType, true),
      StructField("consumption_level_of_baidu", StringType, true),
      StructField("education_level_of_baidu", StringType, true),
      StructField("occupation_of_baidu", StringType, true),
      StructField("frequent_vv_hour", StringType, true),
      StructField("workday_vv_hour", StringType, true),
      StructField("non_workday_vv_hour", StringType, true),
      StructField("app_interest_tag", StringType, true),
      StructField("installed_app_name", StringType, true),
      StructField("frequent_city", StringType, true),
      StructField("mobile_device_manu", StringType, true),
      StructField("device_model", StringType, true),
      StructField("favorite_celebrity", StringType, true),
      StructField("interest_preference_of_baidu", StringType, true),
      StructField("favorite_robert_circle", StringType, true),
      StructField("robert_browsed_circle", StringType, true),
      StructField("realtime_short_content_tags", StringType, true),
      StructField("realtime_short_category_tags", StringType, true),
      StructField("activeuser", StringType, true)
    ))

  def main(args: Array[String]): Unit = {

    val (spark, options) = SparkInit.init(args, "Ranking_xgb_lr")

    import spark.implicits._
    var ratings_orginal = spark.read
      .textFile("viewfs://hadoop/data/robertlee/rec_hive/uid_feed_score/" + options.get.train_date
        + "/part-r-*").rdd.map{
      x =>
        val arr = x.split("::")
        val rating = arr(2).toDouble
        (arr(0), (arr(1).toLong/100).toInt,if(rating>10.0)10.0 else rating)
    }.toDF("deviceId","feedId", "rating")
    println("rating raw data: ")
    ratings_orginal.show(5, false)

    val labels = loadLabelsDataset(spark, options.get.train_date).where("label=1")
      .rdd.map(x=>(x(0).toString, x.getAs[Int]("feed_id")))
        .groupByKey.map(x=>(x._1, x._2.toSeq))
        .toDF("device_id", "label")
    labels.show(false)
    println("In label dataset, deviceid nums: " + labels.select("device_id").distinct().count())

    var ratings = labels
      .join(ratings_orginal, labels.col("device_id") === ratings_orginal.col("deviceId"))
    println("After eval labels join ratings count rows: " + ratings.count)
    ratings = ratings.groupBy("deviceId")
      .agg("feedId" -> "count")
      .filter("count(feedId)>1")
      .select("deviceId").join(ratings, "deviceId")
    println("After removing one feed, samples num: " + ratings.count())

    if(options.get.tuning) cvTuning(labels, ratings_orginal)
    else test(ratings_orginal, ratings, options)
    //cvTuning(ratings)
    //tuning(ratings)
  }

  def test(ratings: DataFrame, test:DataFrame, options: Option[Config]): Unit ={
    val indexer = new StringIndexer()
      .setInputCol("deviceId")
      .setOutputCol("deviceIndex")
      .setHandleInvalid("keep")
      .fit(ratings)

    val als = new ALSExt()
      .setItemCol("feedId")
      .setUserCol("deviceIndex")
      .setMaxIter(options.get.maxIterALS)
      .setRank(options.get.rankALS)
      .setRegParam(options.get.regParamALS)
      .setImplicitPrefs(options.get.IsImplicit)


        val model = als.fit(indexer.transform(ratings))
        //model.transform()
        val predictions = model.transform(indexer.transform(test))
        val evaluator = new RankingEvaluator()
          .setMetricName("precisionAtk")
          .setLabelCol("label")
          .setPredictionCol("recommendations")

        println("precisionAtK: "+ evaluator.evaluate(predictions))

  }

  def cvTuning(labels: DataFrame, orignal: DataFrame): Unit ={
    val training = orignal.join(labels, orignal("deviceId")===labels("device_id"), "left")
    val indexer = new StringIndexer()
      .setInputCol("deviceId")
      .setOutputCol("deviceIndex")
      .setHandleInvalid("keep")

    val als = new ALSExt()
      .setImplicitPrefs(true)
      .setItemCol("feedId")
      .setUserCol("deviceIndex")
      .setMaxIter(5)
      .setRank(5)
    val pipeline = new Pipeline()
      .setStages(Array(indexer, als))

    val paramGrid = new ParamGridBuilder()
      .addGrid(als.rank, Array(5, 10, 15, 20))
      .addGrid(als.maxIter, Array(5, 10, 20, 30, 50))
      //.addGrid(als.implicitPrefs, Array(true, false))
      .build()

    // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    // This will allow us to jointly choose parameters for all Pipeline stages.
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
    // is areaUnderROC.

    val evaluator = new RankingEvaluator()
      .setMetricName("precisionAtk")
      .setLabelCol("label")
      .setPredictionCol("recommendations")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)  // Use 3+ in practice
    // Evaluate up to 2 parameter settings in parallel


    // Run cross-validation, and choose the best set of parameters.
    val cvModel = cv.fit(training)
    println(cvModel.bestModel.params)
  }

  def tuning(training: DataFrame): Unit ={
    val indexer = new StringIndexer()
      .setInputCol("deviceId")
      .setOutputCol("deviceIndex")
      .setHandleInvalid("keep")
    val als = new ALS()
      .setImplicitPrefs(true)
      .setItemCol("feedId")
      .setUserCol("deviceIndex")
      .setMaxIter(5)
      .setRank(5)
    val pipeline = new Pipeline()
      .setStages(Array(indexer, als))

    val paramGrid = new ParamGridBuilder()
      .addGrid(als.rank, Array(5, 10, 15, 20))
      .addGrid(als.maxIter, Array(5, 10, 20, 30, 50))
     // .addGrid(als.implicitPrefs, Array(true, false))
      .build()

    // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    // This will allow us to jointly choose parameters for all Pipeline stages.
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
    // is areaUnderROC.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)  // Use 3+ in practice

    // Run cross-validation, and choose the best set of parameters.
    val cvModel = cv.fit(training)
  }

  def loadLabelsDataset(spark: SparkSession, date: String): DataFrame = {
    import spark.implicits._
    val sdf = new SimpleDateFormat("yyyy-MM-dd")
    val result = new Date(sdf.parse(date).getTime + TimeUnit.DAYS.toMillis(1))
    val date_new = sdf.format(result)
    val path = "viewfs://hadoop/hive/warehouse/robert_harley.db/user_realtime_face_data_dt_new/dt=" + date_new + "/user_realtime_face.txt"
    val raw_data = spark.read.format("csv").schema(SchemaRealtime)
      .option("delimiter", "\t").option("header", "false")
      .load(path)
      .select("device_id", "feed_id", "label")
      .rdd.map { row =>
      val device_id = row.getAs[String]("device_id")
      val feed_id = (row.getAs[String]("feed_id").toLong / 100).toInt
      val label = row.getInt(2)
      (device_id, feed_id, label)
    }.toDF("device_id", "feed_id", "label")
   raw_data
  }

  def colToRows(data: DataFrame): Unit ={
    import data.sparkSession.implicits._
    var predictions = data
      .rdd.flatMap {
      row =>
        val deviceId = row.getAs[Int]("deviceIndex")
        val rec: Seq[Row] = row.getAs[Seq[Row]]("recommendations")
        println(rec)
        rec.map { case Row(feedId: Int, rating: Float) => (deviceId, feedId)}
    }.toDF("deviceId", "feedId")
  }
}
