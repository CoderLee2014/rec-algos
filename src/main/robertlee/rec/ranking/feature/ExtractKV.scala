package robertlee.rec.ranking.feature

import robertlee.rec.ranking.utils.SparkInit
import org.apache.spark.ml.{Model, Pipeline}
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, SchemaUtils}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

class ExtractKV (override val uid: String) extends Model[ExtractKV]
  with HasInputCol with HasOutputCol with DefaultParamsWritable{

  var isExtractKey: Boolean = true// if false, extract value from K-V pair.

  def this() = this(Identifiable.randomUID("ExtractKV"))

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setIsExtractKey(boolean: Boolean): this.type = {
    this.isExtractKey = boolean
    this
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    // schema transformation
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)

    val metadata = transformSchema(dataset.schema)(outputColName).metadata

    // data transformation
    val encode = udf { value: String =>
      if(isExtractKey)
        ExtractKV.getValue(value, 0)
      else
        ExtractKV.getValue(value, 1)
    }
    dataset.select(col("*"), encode(col(inputColName)).as(outputColName, metadata))
  }

  override def transformSchema(schema: StructType): StructType = {
    val col = NumericAttribute.defaultAttr.withName($(outputCol)).toStructField()
    require(!schema.fieldNames.contains(col.name), s"Column ${col.name} already exists.")
    StructType(schema.fields :+ col)
  }

  override def copy(extra: ParamMap): ExtractKV = {
    defaultCopy[ExtractKV](extra).setParent(parent)
  }
}

object ExtractKV{
  def main(args: Array[String]): Unit = {
    val (spark, options) = SparkInit.init(args, "Ranking_GBTFM")
    val data = spark.createDataFrame(
      Seq((8, "bat:0"),
          (0, "4.0:1.0"),
          (1, "5:0.1"),
          (2, "9:-1"),
          (4, "11:23")
      )).toDF("age","name")

    val extract = new ExtractKV()
      .setInputCol("name")
      .setOutputCol("name_value")

    val pipeline = new Pipeline()
    pipeline.setStages(Array(extract))
    val model = pipeline.fit(data)

    model.transform(data).show(false)
  }

  def getValue(v :String, index: Int): Double ={
    try {
      v.split(":")(index).toDouble
    }
    catch {
      case e: Exception => -1
    }
  }
}
