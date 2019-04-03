package org.apache.spark.mllib.optimization

import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
  * Created by vincent on 17-1-4.
  */
class GradientDescentFFM (private var gradient: Gradient, private var updater: Updater, private val param: FFMParameter) extends Optimizer {

  val sgd = true
  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
  private var convergenceTol: Double = 0.001
  /**
    * Set the initial step size of SGD for the first step. Default 1.0.
    * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
    */
  def setStepSize(step: Double): this.type = {
    require(step > 0,
      s"Initial step size must be positive but got ${step}")
    this.stepSize = step
    this
  }

  /**
    * :: Experimental ::
    * Set fraction of data to be used for each SGD iteration.
    * Default 1.0 (corresponding to deterministic/classical gradient descent)
    */
  @Experimental
  def setMiniBatchFraction(fraction: Double): this.type = {
    require(fraction > 0 && fraction <= 1.0,
      s"Fraction for mini-batch SGD must be in range (0, 1] but got ${fraction}")
    this.miniBatchFraction = fraction
    this
  }

  /**
    * Set the number of iterations for SGD. Default 100.
    */
  def setNumIterations(iters: Int): this.type = {
    require(iters >= 0,
      s"Number of iterations must be nonnegative but got ${iters}")
    this.numIterations = iters
    this
  }

  /**
    * Set the regularization parameter. Default 0.0.
    */
  def setRegParam(regParam: Double): this.type = {
    require(regParam >= 0,
      s"Regularization parameter must be nonnegative but got ${regParam}")
    this.regParam = regParam
    this
  }

  /**
    * Set the convergence tolerance. Default 0.001
    * convergenceTol is a condition which decides iteration termination.
    * The end of iteration is decided based on below logic.
    *
    *  - If the norm of the new solution vector is >1, the diff of solution vectors
    *    is compared to relative tolerance which means normalizing by the norm of
    *    the new solution vector.
    *  - If the norm of the new solution vector is <=1, the diff of solution vectors
    *    is compared to absolute tolerance which is not normalizing.
    *
    * Must be between 0.0 and 1.0 inclusively.
    */
  def setConvergenceTol(tolerance: Double): this.type = {
    require(tolerance >= 0.0 && tolerance <= 1.0,
      s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
    this.convergenceTol = tolerance
    this
  }

  /**
    * Set the gradient function (of the loss function of one single data example)
    * to be used for SGD.
    */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
    * Set the updater function to actually perform a gradient step in a given direction.
    * The updater is responsible to perform the update from the regularization term as well,
    * and therefore determines what kind or regularization is used, if any.
    */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }
  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    optimize(data.map(x => (x._1, x._2.toArray.asInstanceOf[Array[FFMNode]])),
      initialWeights, param, true)

  }
  def optimize(data: RDD[(Double, Array[FFMNode])], initialWeights: Vector, param: FFMParameter,
               solver: Boolean): Vector = {
    val (weights, _) = GradientDescentFFM.parallelAdag(data, gradient, initialWeights, param, solver)
    weights
  }

}

object GradientDescentFFM {
  def parallelAdag(
                    data: RDD[(Double, Array[FFMNode])],
                    gradient: Gradient,
                    initialWeights: Vector,
                    param: FFMParameter,
                    solver: Boolean) : (Vector, Array[Double]) = {
    val numIterations = param.n_iters
    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    var weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size
    val slices = data.getNumPartitions


    var converged = false // indicates whether converged based on convergenceTol
    var i = 1
    while (!converged && i < numIterations) {
      val bcWeights = data.context.broadcast(weights)
      // Sample a subset (fraction miniBatchFraction) of the total data
      // compute and sum up the subgradients on this subset (this is one map-reduce)

      val (wSum, lSum) = data.treeAggregate(BDV(bcWeights.value.toArray), 0.0)(
        seqOp = (c, v) => {
          gradient.asInstanceOf[FFMGradient].computeFFM(v._1, (v._2), Vectors.fromBreeze(c._1),
            1.0, param.eta, param.lambda, true, i, solver)
        },
        combOp = (c1, c2) => {
          (c1._1 + c2._1, c1._2 + c2._2)
        }) // TODO: add depth level

      /*
      val (wSum, lSum) = data.treeAggregate(BDV(bcWeights.value.toArray), 0.0)(
        seqOp = (c, v) => {
          computeFFM(v._1, v._2, Vectors.fromBreeze(c._1), 1.0, param.eta, param.lambda, true)
        },
        combOp = (c1, c2) => {
          (c1._1 += c2._1, c1._2 + c2._2)
        }, 7)

      */
      weights = Vectors.dense(wSum.toArray.map(_ / slices))
      stochasticLossHistory += lSum / slices
      println("iter:" + (i + 1) + ",tr_loss:" + lSum / slices)
      i += 1
    }

    (weights, stochasticLossHistory.toArray)

  }

}