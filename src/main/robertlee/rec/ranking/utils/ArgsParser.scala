package robertlee.rec.ranking.utils

object ArgsParser{
    import scopt.OParser
    val builder = OParser.builder[Config]
    val parser = {

          import builder._
          OParser.sequence(
          programName("scopt"),
          head("scopt", "4.x"),
          // option -f, --foo
          opt[String]('d', "train_date")
          .action((x, c) => c.copy( train_date = x))
          .text("data date"),

          opt[String]('h', "data_path")
            .action((x, c) => c.copy( data_path = x))
            .text("data path"),

          opt[String]('x', "version")
            .action((x, c) => c.copy( version = x))
            .text("test version"),

          opt[String]('g', "model_path")
            .action((x, c) => c.copy( model_path = x))
            .text(" spark model path"),

          opt[String]('p', "xgb_pipeline_model_path")
            .action((x, c) => c.copy( xgb_pipeline_model_path = x))
            .text("xgb pipeline model path"),

            opt[String]("fm_model_path")
              .action((x, c) => c.copy( fm_model_path = x))
              .text("fm model path"),

          opt[Boolean]('t', "train_xgb")
            .action((x, c) => c.copy( train_xgb = x))
            .text("if train xgb or load directly"),

            opt[Boolean]("train_lr")
              .action((x, c) => c.copy( train_lr = x))
              .text("if train lr or load directly"),

            opt[Boolean]("train_fm")
              .action((x, c) => c.copy( train_fm = x))
              .text("if train lr or load directly"),

            opt[Boolean]("eval_xgb")
              .action((x, c) => c.copy( eval_xgb = x))
              .text("if eval xgb or not."),

            opt[Boolean]("eval_lr")
              .action((x, c) => c.copy( eval_lr = x))
              .text("if eval lr or not."),

            opt[Boolean]("eval_fm")
              .action((x, c) => c.copy( eval_fm = x))
              .text("if eval fm or not."),

            opt[String]("eval_set")
              .action((x, c) => c.copy( eval_set = x))
              .text("if eval fm or not."),

            opt[Boolean]("real")
              .action((x, c) => c.copy( real = x))
              .text("if eval fm or not."),

            opt[Int]('t', "numIterFM")
              .action((x, c) => c.copy( numIterFM = x))
              .text("iteration num for FM model."),

            opt[Double]('t', "stepSizeFM")
              .action((x, c) => c.copy( stepSizeFM = x))
              .text("stepSize for FM model."),

            opt[String]("regParamFM")
              .action( (s, c) => c.copy( regParamFM = s))
              .text("regParam for FM model."),

            opt[String]("optimizer")
              .action( (s, c) => c.copy( optimizer = s))
              .text("Optimizer for FM model."),

            opt[Int]("task")
              .action( (s, c) => c.copy( task = s))
              .text("Optimizer for FM model."),

            opt[Int]("numCorrections")
              .action( (s, c) => c.copy( numCorrections = s))
              .text("numCorrections for FMWithLBFGS model."),

            opt[Double]("MiniBatchFraction")
              .action( (s, c) => c.copy( MiniBatchFraction = s))
              .text("MiniBatchFraction for FM model."),

            opt[Boolean]('u', "active_users")
              .action((x, c) => c.copy( active_users = x))
              .text("if train using only active_users."),

            opt[Boolean]("tuning")
              .action((x, c) => c.copy( tuning = x))
              .text("if train using tuning for FM."),

            opt[Boolean]("IsImplicit")
              .action((x, c) => c.copy( IsImplicit = x))
              .text("if train using tuning for FM."),

            opt[String]("lr_pipeline_model_path")
              .action((x, c) => c.copy( lr_pipeline_model_path = x))
              .text("Pre-loaded lr model path."),

            opt[String]("query")
              .action((x, c) => c.copy( query = x))
              .text("query condition"),


            opt[Int]('u', "dim")
              .action((x, c) => c.copy( dim = x))
              .text("dim params"),

            opt[Double]("regParamALS")
              .action((x, c) => c.copy( regParamALS = x))
              .text("regParam for ALS"),

            opt[Int]("maxIterALS")
              .action((x, c) => c.copy( maxIterALS = x))
              .text("maxIter for ALS"),

            opt[Int]("rankALS")
              .action((x, c) => c.copy( rankALS = x))
              .text("maxIter for ALS"),

          opt[Int]("days")
            .action((x, c) => c.copy( days = x))
            .text("Days from date to (date - days) for data loading."),

            opt[Int]("eval_days")
              .action((x, c) => c.copy( eval_days = x))
              .text("Days from date to (date - days) for eval data loading."),

              opt[String]("active_node")
              .action((x, c) => c.copy( active_node = x))
              .text("active namenode."),

            opt[String]("eval_date")
              .action((x, c) => c.copy( eval_date = x))
              .text("eval date."),

            opt[String]("itemvec_path")
              .action((x, c) => c.copy( itemvec_path = x))
              .text("item vec data path."),

            opt[String]("uservec_path")
              .action((x, c) => c.copy( uservec_path = x))
              .text("user vec data path.")
            // more options here...
          )
    }

//    // OParser.parse returns Option[Config]
//    val options = OParser.parse(parser1, args, Config()) match {
//      case Some(config) =>
//        // do something
//        println(config.data_path)
//      case _ =>
//      // arguments are bad, error message will have been displayed
//    }
}
case class Config(
                   train_date: String = "",
                   data_path: String = "",
                   days: Int = 7,
                   model_path: String = "viewfs://hadoop/data/robertlee/model/xgb_model_",
                   xgb_model_path: String = "viewfs://hadoop/data/robertlee/model/xgb_model_",
                   xgb_pipeline_model_path: String = "viewfs://hadoop/data/robertlee/model/xgb_pipeline_model",
                   lr_pipeline_model_path: String="",
                   fm_model_path: String="",
                   version: String = "",
                   active_users: Boolean = false,
                   dim: Int=6,
                   numIterFM:Int=100,
                   stepSizeFM:Double=0.15,
                   regParamFM:String="0,0,0",
                   optimizer:String="SGD",
                   task:Int=1,
                   numCorrections:Int=5,
                   MiniBatchFraction:Double=1.0,
                   tuning: Boolean = false,
                   train_xgb: Boolean = false,
                   train_lr: Boolean = true,
                   train_fm: Boolean = true,
                   eval_xgb: Boolean = false,
                   eval_lr: Boolean = true,
                   eval_fm: Boolean = true,
                   real: Boolean = false,
                   eval_set:String="hdfs",
                   query:String="",
                   IsImplicit:Boolean=false,
                   regParamALS:Double=1.0,
                   maxIterALS:Int=10,
                   rankALS:Int=5,
                   eval_days:Int=1,
                   active_node:String="hdfs://namenode02.lhl.hadoop",
                   eval_date:String="",
                   itemvec_path:String="/data/robertlee/itemVec",
                   uservec_path:String="/data/robertlee/userVec"
                 )