package robertlee.rec.ranking.data

import org.apache.spark.sql.types._

object BasicData {



  val schema = new StructType(Array(
    StructField("user_id", StringType, true),
    StructField("feed_id", StringType, true),
    StructField("feature1", DoubleType, true),
    StructField("feature2", IntegerType, true),
    StructField("feature3", BooleanType, true)
  ))

  val schemaTrained_feedSim = new StructType(Array(
    StructField("user_id", StringType, true),
    StructField("feed_id", StringType, true),
    StructField("feature1", DoubleType, true),
    StructField("feature2", IntegerType, true),
    StructField("feature3", BooleanType, true)
  ))

  val schemaTrained_feed2vec = new StructType(Array(
    StructField("user_id", StringType, true),
    StructField("feed_id", StringType, true),
    StructField("feature1", DoubleType, true),
    StructField("feature2", IntegerType, true),
    StructField("feature3", BooleanType, true)
  ))

  val schemaTrained_user2vec = new StructType(Array(
    StructField("user_id", StringType, true),
    StructField("feed_id", StringType, true),
    StructField("feature1", DoubleType, true),
    StructField("feature2", IntegerType, true),
    StructField("feature3", BooleanType, true)
  ))


  val SchemaRealtime = new StructType(
    Array(
      StructField("user_id", StringType, true),
      StructField("feed_id", StringType, true),
      StructField("feature1", DoubleType, true),
      StructField("feature2", IntegerType, true),
      StructField("feature3", BooleanType, true)
    ))
}
