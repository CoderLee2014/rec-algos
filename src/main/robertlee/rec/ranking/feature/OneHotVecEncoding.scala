package robertlee.rec.ranking.feature

import robertlee.rec.ranking.algos.XGBAlgo
import org.apache.spark.ml.Model
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, BinaryAttribute}
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasInputCols, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.StructType

class OneHotVecEncoding (override val uid: String) extends Model[OneHotVecEncoding]
  with HasInputCol with HasOutputCol with DefaultParamsWritable{

  var valuesSize: Int  = _
  var dim: Int = _

  def this() = this(Identifiable.randomUID("OneHotVecEncoding"))

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setValuesSize(valueSize: Int): this.type ={
    this.valuesSize = valueSize
    this
  }

  def setDim(dim: Int): this.type ={
    this.dim = dim
    this
  }


  override def transform(dataset: Dataset[_]): DataFrame = {
    // schema transformation
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)

    val outputAttrGroupFromSchema = AttributeGroup.fromStructField(
      transformSchema(dataset.schema)(outputColName))

    val outputAttrGroup = if (outputAttrGroupFromSchema.size < 0) {
      val numAttrs = valuesSize * dim
      val outputAttrNames = Array.tabulate(numAttrs)(_.toString)
      val outputAttrs: Array[Attribute] =
        outputAttrNames.map(name => BinaryAttribute.defaultAttr.withName(name))
      new AttributeGroup(outputColName, outputAttrs)
    } else {
      outputAttrGroupFromSchema
    }

    val metadata = outputAttrGroup.toMetadata()

    // data transformation
    val encode = udf { row: Row =>
      OneHotVecEncoding.oneHotLeafPred(row, this.valuesSize)
    }

    dataset.select(col("*"), encode(dataset(inputColName)).as(outputColName, metadata))
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

  override def copy(extra: ParamMap): OneHotVecEncoding = {
    defaultCopy[OneHotVecEncoding](extra).setParent(parent)
  }
}

object OneHotVecEncoding{
  def oneHotLeafPred(row: Row, valuesSize: Int): Vector ={
    var pred_leaf = row.getAs[Seq[Float]]("predLeaf").map(_.toInt).toArray
    val num_leaf = Math.pow(2, valuesSize + 1).toInt
    try{
      var i = -1
      pred_leaf.foreach{
        x =>
          i += 1
          pred_leaf(i) = x + i * num_leaf
      }
    }catch {
      case e: Exception =>  println("get xgb_leaf exception.")
    }
    Vectors.sparse(pred_leaf.length*num_leaf, pred_leaf,new Array[Double](pred_leaf.length).map(_ => 1.0))
  }

  def main(args: Array[String]): Unit = {

  }
}
