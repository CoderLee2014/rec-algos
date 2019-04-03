package org.apache.spark.ml
import ml.combust.mleap.core.feature.StringMapModel
import ml.dmlc.xgboost4j.scala.DMatrix
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types._
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel
import org.apache.spark.mllib.linalg.SparseVector

class PredictLeaf (override val uid: String,
                val model:  XGBoostClassificationModel) extends Transformer
  with HasInputCol
  with HasOutputCol {
 // def this(model: StringMapModel) = this(uid = Identifiable.randomUID("predict_leaf"), model = model)

  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  @org.apache.spark.annotation.Since("2.0.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    val broadcastBooster = dataset.sparkSession.sparkContext.broadcast(model.nativeBooster)
    val getPredLeaf = udf {
      (features: SparseVector) =>
        val features_dense = features.toDense
        var pred_leaf = broadcastBooster.value
          .predictLeaf(new DMatrix(features_dense.values.map(_.toFloat), 1, features_dense.values.length))(0)
          .map(_.toInt)
        val num_leaf = Math.pow(2, model.getMaxDepth + 1).toInt
        var i = 0
        pred_leaf.foreach{
          x =>
            i += 1
            pred_leaf(i) = x + i * num_leaf
        }
        pred_leaf
    }
    dataset.withColumn($(outputCol), getPredLeaf(dataset($(inputCol))))
  }

  override def copy(extra: ParamMap): Transformer = copyValues(new PredictLeaf(uid, model), extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    require(schema($(inputCol)).dataType.isInstanceOf[StringType],
      s"Input column must be of type StringType but got ${schema($(inputCol)).dataType}")
    val inputFields = schema.fields
    require(!inputFields.exists(_.name == $(outputCol)),
      s"Output column ${$(outputCol)} already exists.")

    StructType(schema.fields :+ StructField($(outputCol), DoubleType))
  }

}
