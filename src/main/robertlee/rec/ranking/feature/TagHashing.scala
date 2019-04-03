package robertlee.rec.ranking.feature

import robertlee.rec.ranking.utils.SparkInit
import org.apache.spark.ml.{Model, Pipeline}
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.{DoubleType, NumericType, StringType, StructType}

import scala.util.control.Breaks.{break, breakable}

class TagHashing(override val uid: String) extends Model[TagHashing]
  with HasInputCol with HasOutputCol with DefaultParamsWritable{

  var dim: Int = 1000
  var topK: Int = 100
  var seeds: Int = 12345
  var filterPosTag: Boolean = false

  def this() = this(Identifiable.randomUID("TagHashing"))

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setHashingDim(dim: Int): this.type = {
    this.dim = dim
    this
  }

  def setTopKTags(k: Int): this.type = {
    this.topK = k
    this
  }

  def setSeeds(seed: Int): this.type = {
    this.seeds = seed
    this
  }

  def setFilterPosTags(filter: Boolean): this.type = {
    this.filterPosTag = filter
    this
  }

  override def copy(extra: ParamMap): TagHashing = {
    defaultCopy[TagHashing](extra).setParent(parent)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    // schema transformation
    val inputColName = $(inputCol)
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
    val encode = udf { value: String =>
      if(this.filterPosTag){
        TagHashing.hashing(TagHashing.getPosTags(value, inputColName, this.topK),this.dim, this.seeds)
      }else
      TagHashing.hashing(value, this.dim, this.seeds).toSparse
    }
    dataset.select(col("*"), encode(col(inputColName)).as(outputColName, metadata))
  }

  override def transformSchema(schema: StructType): StructType = {
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)
    val inputFields = schema.fields

    require(schema(inputColName).dataType.isInstanceOf[StringType],
      s"Input column must be of type StringType but got ${schema(inputColName).dataType}")
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
          s"The input column $inputColName cannot be numeric.")
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

object TagHashing{
  def main(args: Array[String]): Unit = {
    val (spark, options) = SparkInit.init(args, "Ranking_GBTFM")
    val data = spark.createDataFrame(Seq((8, "bat"),
      (0, "mouse:1.0,daf:2.0"),
      (1, "horse:0.1"),
      (2, "horse:-1"),
      (4, "horse:23")
    )).toDF("age","name")

    val hasher = new TagHashing()
      .setInputCol("name")
      .setOutputCol("name_vec")
      .setHashingDim(10)
      .setTopKTags(10)

    val pipeline = new Pipeline()
    pipeline.setStages(Array(hasher))
    val model = pipeline.fit(data)

    model.transform(data).show(false)
  }

  def hashing(value: String, bucket_size:Int, seed: Int): DenseVector = {
    val words: Array[String] = value.split(",")
    var buckets = new Array[Double](bucket_size)
    try{
      breakable {
        for (word <- words) {
          val term = word.split(":")(0)
          if (term.equals("null")|| term.equals("")) {
            break
          }
          val bucket = murmurHash(term, seed) % bucket_size
          buckets(bucket) = 1.0
        }
      }
    }catch{
      case e : Exception =>
        println("hw exp: ", e)
        println("hw words: ", words)
        buckets = buckets.map(_ => 0.0)
    }
    new DenseVector(buckets)
  }

  def murmurHash(word: String, seed: Int): Int = {
    val c1 = 0xcc9e2d51
    val c2 = 0x1b873593
    val r1 = 15
    val r2 = 13
    val m = 5
    val n = 0xe6546b64

    var hash = seed //12345

    for (ch <- word.toCharArray) {
      var k = ch.toInt
      k = k * c1
      k = (k << r1) | (hash >> (32 - r1))
      k = k * c2

      hash = hash ^ k
      hash = (hash << r2) | (hash >> (32 - r2))
      hash = hash * m + n
    }

    hash = hash ^ word.length
    hash = hash ^ (hash >> 16)
    hash = hash * 0x85ebca6b
    hash = hash ^ (hash >> 13)
    hash = hash * 0xc2b2ae35
    hash = hash ^ (hash >> 16)

    hash
  }

  def getPosTags(value: String, field_name: String, top: Int): String ={
    try {
      value.split(",").filter(PosTagFiltering).take(top).map(x => x.split(":")(0)).mkString(",")
    }catch{
      case e: Exception =>
        println(s"${field_name} processing exp: ", value);
        "null"
    }
  }

  def PosTagFiltering(x: String): Boolean ={
    var cond = false;
    try {
      if (x.split(":")(1).toDouble >= 0.0)
        cond = true
    }
    catch {
      case e: Exception =>
        if(x.split(":")(0).equals("null") || x.split(":")(0).equals(""))
          cond = false
        else
          cond = true // tags without scores.
    };
    cond
  }
}
