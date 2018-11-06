/**
 * @file ma_ss.hpp
 * @author Marcus Edel
 *
 * MaSS (Momentum-added Stochastic Solver), an accelerated SGD method for
 * optimizing over-parametrized networks.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_MA_SS_MA_SS_HPP
#define ENSMALLEN_MA_SS_MA_SS_HPP

namespace ens {

/**
 * MaSS (Momentum-added Stochastic Solver) is an accelerated SGD method for
 * optimizing over-parametrized networks.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Liu2018,
 *   author = {{Liu}, C. and {Belkin}, M.},
 *   title = "{MaSS: an Accelerated Stochastic Method for Over-parametrized
 *             Learning}",
 *   journal = {ArXiv e-prints},
 *   year    = {2018},
 *   url     = {https://arxiv.org/abs/1810.13395}
 * }
 *
 * For MaSS to work, a DecomposableFunctionType template parameter is required.
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
class MaSS
{
 public:
  /**
   * Construct the MaSS (Momentum-added Stochastic Solver) optimizer with the
   * given function and parameters. The defaults here are not necessarily good
   * for the given problem, so it is suggested that the values used be tailored
   * to the task at hand.  The maximum number of iterations refers to the
   * maximum number of points that are processed (i.e., one iteration equals
   * one point; one iteration does not equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param secondaryStepSize Secondary step size for each iteration.
   * @param batchSize Number of points to process in a single step.
   * @param acceleration Acceleration parameter.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *        limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *        function is visited in linear order.
   */
  MaSS(const double stepSize = 0.01,
       const double secondaryStepSize = 0.083,
       const size_t batchSize = 32,
       const double acceleration = 0.01,
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

  //! Get the secondary step size.
  double SecondaryStepSize() const { return secondaryStepSize; }
  //! Modify the secondary step size.
  double& SecondaryStepSize() { return secondaryStepSize; }

  //! Get the batch size.
  size_t BatchSize() const { return batchSize; }
  //! Modify the batch size.
  size_t& BatchSize() { return batchSize; }

  //! Get the acceleration parameter.
  double Acceleration() const { return acceleration; }
  //! Modify the acceleration parameter.
  double& Acceleration() { return acceleration; }

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

  //! The secondary step size for each example.
  double secondaryStepSize;

  //! The batch size for processing.
  size_t batchSize;

  //! The acceleration parameter.
  double acceleration;

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
#include "ma_ss_impl.hpp"

#endif
