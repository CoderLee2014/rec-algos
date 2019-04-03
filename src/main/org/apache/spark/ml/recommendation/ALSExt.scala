package org.apache.spark.ml.recommendation

import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.annotation.{DeveloperApi, Since}
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, Instrumentation, MLWritable}
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.types.FloatType
import org.apache.spark.storage.StorageLevel

class ALSExt(override val uid: String)
  extends ALS(uid) with ALSParams
    with DefaultParamsWritable{

  def this() = this(Identifiable.randomUID("alsExt"))

  override def fit(dataset: Dataset[_]): ALSModelExt = {
    transformSchema(dataset.schema)
    import dataset.sparkSession.implicits._

    val r = if ($(ratingCol) != "") col($(ratingCol)).cast(FloatType) else lit(1.0f)
    val ratings = dataset
      .select(checkedCast(col($(userCol))), checkedCast(col($(itemCol))), r)
      .rdd
      .map { row =>
        Rating(row.getInt(0), row.getInt(1), row.getFloat(2))
      }

    val instr = Instrumentation.create(this, ratings)
    instr.logParams(rank, numUserBlocks, numItemBlocks, implicitPrefs, alpha, userCol,
      itemCol, ratingCol, predictionCol, maxIter, regParam, nonnegative, checkpointInterval,
      seed, intermediateStorageLevel, finalStorageLevel)

    val (userFactors, itemFactors) = ALS.train(ratings, rank = $(rank),
      numUserBlocks = $(numUserBlocks), numItemBlocks = $(numItemBlocks),
      maxIter = $(maxIter), regParam = $(regParam), implicitPrefs = $(implicitPrefs),
      alpha = $(alpha), nonnegative = $(nonnegative),
      intermediateRDDStorageLevel = StorageLevel.fromString($(intermediateStorageLevel)),
      finalRDDStorageLevel = StorageLevel.fromString($(finalStorageLevel)),
      checkpointInterval = $(checkpointInterval), seed = $(seed))
    val userDF = userFactors.toDF("id", "features")
    val itemDF = itemFactors.toDF("id", "features")
    val model = new ALSModelExt(uid, $(rank), userDF, itemDF).setParent(this)
    instr.logSuccess(model)
    copyValues(model)
  }
  override def copy(extra: ParamMap): ALSExt = defaultCopy(extra)

}
class ALSModelExt(
       override val uid: String,
       override val rank: Int,
      @transient override val userFactors: DataFrame,
      @transient override val itemFactors: DataFrame )
  extends ALSModel(uid, rank, userFactors, itemFactors)
    with ALSModelParams with MLWritable{

  override def transform(dataset: Dataset[_]): DataFrame = {
    import dataset.sparkSession.implicits._
    var predictions = ExtMatrixFactorizationModelHelper.recommendProductsForUsers(this, 100, 100000)
      .toDF("deviceIndex", "recommendations")
      .map{
        row =>
          val deviceId = row.getAs[Int]("deviceIndex").toDouble
          val rec: Seq[Row] = row.getAs[Seq[Row]]("recommendations")
          (deviceId, rec.map(row=>row.getAs[Int]("feedId")))
      }.toDF("deviceIdIndex", "recommendations")
    predictions = predictions
      .join(dataset.select("deviceIndex","label").dropDuplicates(), predictions("deviceIdIndex")===dataset("deviceIndex"))
        .na.drop()
    predictions.show(10,false)
    predictions.select("recommendations","label").dropDuplicates()
  }

  override def setParent(parent: Estimator[ALSModel]): ALSModelExt = {
    this.parent = parent
    this.asInstanceOf[ALSModelExt]
  }

  override def copy(extra: ParamMap): ALSModelExt = {
    val copied = new ALSModelExt(uid, rank, userFactors, itemFactors)
    copyValues(copied, extra).setParent(parent)
  }
}