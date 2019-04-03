package robertlee.rec.ranking.algos

import robertlee.rec.ranking.utils.Config
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, SparkSession}

abstract class Algo {
  def init(options:Option[Config], dataSchema: StructType):Unit
  def fit(data: DataFrame, options: Option[Config]): PipelineModel
  def transform(data:DataFrame, options: Option[Config]):DataFrame
  def evaluate(spark: SparkSession, options: Option[Config]):Unit
  def loadModel(options: Option[Config]):Unit
  def buildPipeline():Pipeline
}

