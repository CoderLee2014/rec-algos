package robertlee.rec.ranking.feature

import robertlee.rec.ranking.utils.SparkInit
import org.apache.spark.ml.{Model, Pipeline, Transformer}
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{DoubleType, NumericType, StructField, StructType}


class OneHotEncoding(override val uid: String) extends Model[OneHotEncoding]
  with HasInputCol with HasOutputCol with DefaultParamsWritable{

  final val dropLast = false

  var splits: Array[String] = _

  def this() = this(Identifiable.randomUID("oneHot"))

  def setSplits(splits: Array[String]): this.type ={
      this.splits = splits
      this
  }
  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    // schema transformation
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)

    val outputAttrGroupFromSchema = AttributeGroup.fromStructField(
      transformSchema(dataset.schema)(outputColName))

    val outputAttrGroup = if (outputAttrGroupFromSchema.size < 0) {
      val numAttrs = splits.size
      val outputAttrNames = Array.tabulate(numAttrs)(_.toString)
      val outputAttrs: Array[Attribute] =
        outputAttrNames.map(name => BinaryAttribute.defaultAttr.withName(name))
      new AttributeGroup(outputColName, outputAttrs)
    } else {
      outputAttrGroupFromSchema
    }

    val metadata = outputAttrGroup.toMetadata()

    // data transformation
    val encode = udf { value: String =>
      new DenseVector(OneHotEncoding.oneHotEncoding(splits, value)).toSparse
    }

    dataset.select(col("*"), encode(col(inputColName)).as(outputColName, metadata))
  }

  override def copy(extra: ParamMap): OneHotEncoding = {
    defaultCopy[OneHotEncoding](extra).setParent(parent)
  }

  override def transformSchema(schema: StructType): StructType = {
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)
    val inputFields = schema.fields

    require(!inputFields.exists(_.name == outputColName),
      s"Output column $outputColName already exists.")

    val outputAttrGroup = new AttributeGroup($(outputCol))

    val outputFields = inputFields :+ outputAttrGroup.toStructField()
    StructType(outputFields)
  }

}

object OneHotEncoding{

  def main(args: Array[String]): Unit = {
    val (spark, options) = SparkInit.init(args, "Ranking_GBTFM")
    val data = spark.createDataFrame(Seq((8, "bat"),
      (0, "mouse"),
      (1, "horse"),
      (2, "horse"),
      (4, "horse")
    )).toDF("age","name")

    val splits = Array[Double](0,1,2).map(_.toString)
    val encoder = new OneHotEncoding()
      .setInputCol("age")
      .setSplits(splits)
        .setOutputCol("age_vec")

    val pipeline = new Pipeline()
    pipeline.setStages(Array(encoder))
    val model = pipeline.fit(data)
    model.transform(data).show(false)
    encoder.transform(data)
  }

  def oneHotEncoding(split: Array[String], value: String): Array[Double] ={
    var i = -1
    var arr = new Array[Double](split.size).map(_=>0.0)
    split.foreach{
      threshold =>
        if(value.equals(threshold))
          i += 1
    }
    if(i>=0 && i<arr.size)
      arr(i) = 1
    arr
  }
}
