import java.net.URI

import ml.combust.bundle.BundleFile
import ml.combust.bundle.serializer.SerializationFormat
import org.apache.spark.ml.feature.{OneHotEncoder, OneHotEncoderModel, StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.mleap.SparkUtil
import ml.combust.mleap.spark.SparkSupport._

import resource._
object Test{

  def main(args: Array[String]): Unit = {
    // Create a sample pipeline that we will serialize
    // And then deserialize using various formats
//
//    val conf = new SparkConf().set("spark.hadoop.validateOutputSpecs", "false")
//    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//    //conf.registerKryoClasses(Array(classOf[MyClass1],classOf[MyClass2]))
//    val sc = new SparkContext(conf)
//    val spark = SparkSession.builder().appName("Ranking_xgb_lr").getOrCreate()
//
//
//    val df = spark.createDataFrame(
//      Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
//    ).toDF("id", "a_string")

    val stringIndexer = new StringIndexerModel(labels = Array("Hello, MLeap!", "Another row")).
      setInputCol("a_string").
      setOutputCol("a_string_index")
//    val onehot = new OneHotEncoder()
//      .setInputCols(Array("a_string_index"))
//      .setOutputCols(Array("a_string_onehot"))
    val featureAssembler = new VectorAssembler().setInputCols(Array("a_string_index")).
      setOutputCol("features")

    // create pipeline that will transform data
    val pipeline = SparkUtil.createPipelineModel(Array(stringIndexer, featureAssembler))


//    // Create an implicit custom mleap context for saving/loading
//    implicit val sbc = MleapContext.defaultContext.copy(
//      registry = MleapContext.defaultContext.bundleRegistry.registerFileSystem(bundleFs)
//    )
// Use a URI to locate the bundle
//    val bundleUri = new URI("/tmp/test.bundle.zip")
//    pipeline.writeBundle.save(bundleUri)

    println("Test")
    // write to tmp storage
    for(bundle <- managed(BundleFile("jar:file:/tmp/simple.json.zip"))) {
      pipeline.writeBundle.format(SerializationFormat.Json).save(bundle).get
    }
    println("finish")
  }
}
