
package robertlee.rec.ranking.algos

import org.apache.spark.sql._
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix}
import ml.dmlc.xgboost4j.java.Rabit
import org.apache.spark.ml.linalg.{DenseVector => MLDenseVector, Vector => MLVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{ArrayType, FloatType}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.spark.TaskContext
import scala.collection.JavaConverters._
import org.apache.spark.ml.linalg.{SparseVector, Vectors, Vector}

object XGBUtils
{

    def transformLeaf(booster: Booster, testSet: Dataset[_],spark:SparkSession): DataFrame = {
      val predLeafRDD = produceRowRDD(booster, testSet, predLeaf = true, spark=spark)
      predLeafRDD.take(2)
      testSet.sparkSession.createDataFrame(predLeafRDD, testSet.schema.add("predLeaf",
        ArrayType(FloatType, containsNull = false) ))
    }

   def produceRowRDD(booster: Booster, testSet: Dataset[_], outputMargin: Boolean = false,
                                predLeaf: Boolean = false, spark: SparkSession): RDD[Row] = {
     import spark.implicits._
      val broadcastBooster = testSet.sparkSession.sparkContext.broadcast(booster)
      val appName = testSet.sparkSession.sparkContext.appName
     try {
       testSet.rdd.mapPartitions{
         rowIterator =>
           if (rowIterator.hasNext) {
             val rabitEnv = Array("DMLC_TASK_ID" -> TaskContext.getPartitionId().toString).toMap
             Rabit.init(rabitEnv.asJava)
             val (rowItr1, rowItr2) = rowIterator.duplicate
             val vectorIterator = rowItr2.map(row => row.asInstanceOf[Row].getAs[MLVector]("features")).toList.iterator
             val testDataset = new DMatrix(vectorIterator.map {
               v =>
                 v match {
                   case v: MLDenseVector =>
                     XGBLabeledPoint(0.0f, null, v.values.map(_.toFloat))
                   case v: SparseVector =>
                     XGBLabeledPoint(0.0f, v.indices, v.values.map(_.toFloat))
                 }
             })
             try {
               val rawPredictResults = {
                 if (!predLeaf) {
                   broadcastBooster.value.predict(testDataset, outputMargin).map(Row(_)).iterator
                 } else {
                   broadcastBooster.value.predictLeaf(testDataset).map(Row(_)).iterator
                 }
               }
               Rabit.shutdown()
               // concatenate original data partition and predictions
               rowItr1.zip(rawPredictResults).map {
                 case (originalColumns: Row, predictColumn: Row) =>
                   Row.fromSeq(originalColumns.toSeq ++ predictColumn.toSeq)
               }
             } finally {
               testDataset.delete()
             }
           } else {
             Iterator[Row]()
           }
       }
     }catch{
       case exception: Exception =>
         spark.sparkContext.emptyRDD[Row]
     }
    }

  def oneHotLeafPred(row: Row): Vector ={
    var pred_leaf = row.getAs[Seq[Float]]("predLeaf").map(_.toInt).toArray
    val num_leaf = Math.pow(2, XGBAlgo.maxDepth + 1).toInt
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
}