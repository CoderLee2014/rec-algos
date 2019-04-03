package robertlee.rec.ranking.feature

import robertlee.rec.ranking.feature.TagHashing.PosTagFiltering
import robertlee.rec.ranking.utils.SparkInit
import org.apache.spark.ml.{Model, Pipeline}
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.{DenseVector, VectorUDT}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasInputCols, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{col, struct, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types._

import scala.util.control.Breaks.{break, breakable}

class TagUnionHashing(override val uid: String) extends Model[TagUnionHashing]
with HasInputCols with HasOutputCol with DefaultParamsWritable{
  var dim: Int = 1000
  var topK: Array[Int] = _
  var seeds: Int = 12345
  var filterPosTag: Array[Boolean] = _

  def this() = this(Identifiable.randomUID("TagUnionHashing"))

  def setInputCols(values: Array[String]): this.type = set(inputCols, values)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setHashingDim(dim: Int): this.type = {
    this.dim = dim
    this
  }

  def setTopKsTags(ks: Array[Int]): this.type = {
    this.topK = ks
    this
  }

  def setSeeds(seed: Int): this.type = {
    this.seeds = seed
    this
  }

  def setFilterPosTags(filters: Array[Boolean]): this.type = {
    this.filterPosTag = filters
    this
  }

  override def copy(extra: ParamMap): TagUnionHashing = {
    defaultCopy[TagUnionHashing](extra).setParent(parent)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    // schema transformation
    val inputColsName = $(inputCols)
    val outputColName = $(outputCol)

    val outputAttrGroupFromSchema = AttributeGroup.fromStructField(
      transformSchema(dataset.schema)(outputColName))

    val outputAttrGroup = if (outputAttrGroupFromSchema.size < 0) {
      val numAttrs = this.dim
      val outputAttrNames = Array.tabulate(numAttrs)(_.toString)
      val outputAttrs: Array[Attribute] =
        outputAttrNames.map(name => BinaryAttribute.defaultAttr.withName(name))
      new AttributeGroup(outputColName, outputAttrs)
    } else {
      outputAttrGroupFromSchema
    }

    val metadata = outputAttrGroup.toMetadata()

    // data transformation
    val encode = udf { r: Row =>
      val value = $(inputCols).zipWithIndex.map {
        case(c, i)=>
          if(filterPosTag(i))
            TagUnionHashing.getPosTags(r.getAs[String](c), c, topK(i))
          else
            TagUnionHashing.getTags(r.getAs[String](c), c)
      }.flatMap{case x: Array[String] => x}.toSet.filter(x => !(x.equals("null"))).mkString(",")
      TagHashing.hashing(value, this.dim, this.seeds).toSparse
    }
    val args = inputColsName.map { c => dataset(c)}
    dataset.select(col("*"), encode(struct(args: _*)).as(outputColName, metadata))
  }

  override def transformSchema(schema: StructType): StructType = {
    val inputColsName = $(inputCols)
    val outputColName = $(outputCol)
    val inputFields = schema.fields

    require(!inputFields.exists(_.name == outputColName),
      s"Output column $outputColName already exists.")
    val inputAttr = NominalAttribute.defaultAttr
    val outputAttrNames: Option[Array[String]] = inputAttr match {
      case nominal: NominalAttribute =>
        if (nominal.values.isDefined) {
          nominal.values
        } else if (nominal.numValues.isDefined) {
          nominal.numValues.map(n => Array.tabulate(n)(_.toString))
        } else {
          None
        }
      case binary: BinaryAttribute =>
        if (binary.values.isDefined) {
          binary.values
        } else {
          Some(Array.tabulate(2)(_.toString))
        }
      case _: NumericAttribute =>
        throw new RuntimeException(
          s"The input column $inputColsName cannot be numeric.")
      case _ =>
        None // optimistic about unknown attributes
    }

    val outputAttrGroup = if (outputAttrNames.isDefined) {
      val attrs: Array[Attribute] = outputAttrNames.get.map { name =>
        NumericAttribute.defaultAttr.withName(name)
      }
      new AttributeGroup($(outputCol), attrs)
    } else {
      new AttributeGroup($(outputCol))
    }

    val outputFields = inputFields :+ outputAttrGroup.toStructField()
    StructType(outputFields)
  }
}

object TagUnionHashing{
  def main(args: Array[String]): Unit = {
    val (spark, options) = SparkInit.init(args, "Ranking_GBTFM")
    val data = spark.createDataFrame(Seq((8, "bat","mouse"),
      (0, "mouse:1.0,daf:2.0","horse"),
      (1, "horse:0.1","rabit"),
      (2, "horse:-1", "monkey"),
      (4, "horse:23", "kitty")
    )).toDF("age","name", "tags")

    val hasher = new TagUnionHashing()
      .setInputCols(Array("name", "tags"))
      .setOutputCol("name_vec")
      .setFilterPosTags(Array(true, false))
      .setTopKsTags(Array(10,10))
      .setHashingDim(10)

    val pipeline = new Pipeline()
    pipeline.setStages(Array(hasher))
    val model = pipeline.fit(data)

    model.transform(data).show(false)
  }

  def getTags(value: String, field_name: String): Array[String] ={
    try {
      value.split(",").map(x => x.split(":")(0))
    }catch{
      case e: Exception =>
        println(s"${field_name} processing exp: ", value);
        Array("null")
    }
  }

  def getPosTags(value: String, field_name: String, top: Int): Array[String] ={
    try {
      value.split(",").filter(PosTagFiltering).take(top).map(x => x.split(":")(0))
    }catch{
      case e: Exception =>
        println(s"${field_name} processing exp: ", value);
        Array("null")
    }
  }
}
