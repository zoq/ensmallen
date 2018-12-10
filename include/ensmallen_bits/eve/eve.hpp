/**
 * @file eve.hpp
 * @author Marcus Edel
 *
 * Eve: a gradient based optimization method with Locally
 * and globally adaptive learning rates.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_EVE_EVE_HPP
#define ENSMALLEN_EVE_EVE_HPP

namespace ens {

/**
 * Eve is a stochastic gradient based optimization method with locally and
 * globally adaptive learning rates. Stochastic Gradient Descent is a
 * technique for minimizing a function which can be expressed as a sum of other
 * functions.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Koushik2016,
 *   author  = {Jayanth Koushik and Hiroaki Hayashi},
 *   title   = {Improving Stochastic Gradient Descent with Feedback},
 *   journal = {CoRR},
 *   year    = {2016},
 *   url     = {http://arxiv.org/abs/1611.01505}
 * }
 *
 * For EVE to work, a DecomposableFunctionType template parameter is required.
 * This class must implement the following function:
 *
 *   size_t NumFunctions();
 *   double Evaluate(const arma::mat& coordinates,
 *                   const size_t i,
 *                   const size_t batchSize);
 *   void Gradient(const arma::mat& coordinates,
 *                 const size_t i,
 *                 arma::mat& gradient,
 *                 const size_t batchSize);
 *
 * NumFunctions() should return the number of functions (\f$n\f$), and in the
 * other two functions, the parameter i refers to which individual function (or
 * gradient) is being evaluated.  So, for the case of a data-dependent function,
 * such as NCA, NumFunctions() should return the number of points in the
 * dataset, and Evaluate(coordinates, 0) will evaluate the objective function on
 * the first point in the dataset (presumably, the dataset is held internally in
 * the DecomposableFunctionType).
 */
class EVE
{
 public:
  /**
   * Construct the EVE optimizer with the given function and parameters. The
   * defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param batchSize Number of points to process in a single step.
   * @param beta1 Exponential decay rate for the first moment estimates.
   * @param beta2 Exponential decay rate for the weighted infinity norm
            estimates.
   * @param epsilon Value used to initialise the mean squared gradient parameter.
   * @param clip Clipping range to avoid extreme valus.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *        limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *        function is visited in linear order.
   */
  EVE(const double stepSize = 0.001,
      const size_t batchSize = 32,
      const double beta1 = 0.9,
      const double beta2 = 0.999,
      const double beta3 = 0.999,
      const double epsilon = 1e-8,
      const double clip = 10,
      const size_t maxIterations = 100000,
      const double tolerance = 1e-5,
      const bool shuffle = true);

  /**
   * Optimize the given function using stochastic gradient descent.  The given
   * starting point will be modified to store the finishing point of the
   * algorithm, and the final objective value is returned.
   *
   * @tparam DecomposableFunctionType Type of the function to be optimized.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename DecomposableFunctionType>
  double Optimize(DecomposableFunctionType& function, arma::mat& iterate);

  //! Get the step size.
  double StepSize() const { return stepSize; }
  //! Modify the step size.
  double& StepSize() { return stepSize; }

  //! Get the batch size.
  size_t BatchSize() const { return batchSize; }
  //! Modify the batch size.
  size_t& BatchSize() { return batchSize; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

  //! Get whether or not the individual functions are shuffled.
  bool Shuffle() const { return shuffle; }
  //! Modify whether or not the individual functions are shuffled.
  bool& Shuffle() { return shuffle; }

 private:
  //! The step size for each example.
  double stepSize;

  //! The batch size for processing.
  size_t batchSize;

  //! The smoothing parameter.
  double beta1;

  //! The second moment coefficient.
  double beta2;

  //! The third moment coefficient.
  double beta3;

  //! The epsilon value used to initialise the squared gradient parameter.
  double epsilon;

  //! The clip value used to clip the term to avoid extreme values.
  double clip;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;

  //! Controls whether or not the individual functions are shuffled when
  //! iterating.
  bool shuffle;
};

} // namespace ens

// Include implementation.
#include "eve_impl.hpp"

#endif