/**
 * @file ma_ss_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of MaSS (Momentum-added Stochastic Solver).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_MA_SS_MA_SS_IMPL_HPP
#define ENSMALLEN_MA_SS_MA_SS_IMPL_HPP

// In case it hasn't been included yet.
#include "ma_ss.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

inline MaSS::MaSS(const double stepSize,
                  const double secondaryStepSize,
                  const size_t batchSize,
                  const double acceleration,
                  const size_t maxIterations,
                  const double tolerance,
                  const bool shuffle) :
    stepSize(stepSize),
    secondaryStepSize(secondaryStepSize),
    batchSize(batchSize),
    acceleration(acceleration),
    maxIterations(maxIterations),
    tolerance(tolerance),
    shuffle(shuffle)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename DecomposableFunctionType>
double MaSS::Optimize(DecomposableFunctionType& function, arma::mat& iterate)
{
  typedef Function<DecomposableFunctionType> FullFunctionType;
  FullFunctionType& f(static_cast<FullFunctionType&>(function));

  // Make sure we have all the methods that we need.
  traits::CheckDecomposableFunctionTypeAPI<FullFunctionType>();

  // Find the number of functions to use.
  const size_t numFunctions = f.NumFunctions();

  // To keep track of where we are and how things are going.
  size_t currentFunction = 0;
  double overallObjective = 0;
  double lastOverallObjective = DBL_MAX;

  double objective = 0;
  double lastObjective = 0;

  arma::mat u = iterate;

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  const size_t actualMaxIterations = (maxIterations == 0) ?
      std::numeric_limits<size_t>::max() : maxIterations;
  for (size_t i = 0; i < actualMaxIterations; /* incrementing done manually */)
  {
    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0 && i > 0)
    {
      // Output current objective function.
      Info << "MaSS: iteration " << i << ", objective " << overallObjective
          << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Warn << "MaSS: converged to " << overallObjective << "; terminating"
            << " with failure.  Try a smaller step size?" << std::endl;
        return overallObjective;
      }

      if (std::abs(lastOverallObjective - overallObjective) < tolerance)
      {
        Info << "MaSS: minimized within tolerance " << tolerance << "; "
            << "terminating optimization." << std::endl;
        return overallObjective;
      }

      // Reset the counter variables.
      lastOverallObjective = overallObjective;
      overallObjective = 0;
      currentFunction = 0;

      if (shuffle) // Determine order of visitation.
        f.Shuffle();
    }

    // Find the effective batch size; we have to take the minimum of three
    // things:
    // - the batch size can't be larger than the user-specified batch size;
    // - the batch size can't be larger than the number of iterations left
    //       before actualMaxIterations is hit;
    // - the batch size can't be larger than the number of functions left.
    const size_t effectiveBatchSize = std::min(
        std::min(batchSize, actualMaxIterations - i),
        numFunctions - currentFunction);

    // Technically we are computing the objective before we take the step, but
    // for many FunctionTypes it may be much quicker to do it like this.
    objective = f.EvaluateWithGradient(u, currentFunction,
        gradient, effectiveBatchSize);
    overallObjective += objective;

    iterate = u - stepSize * gradient;

    if (i < (actualMaxIterations - 1))
    {
      u = (1 + acceleration) * iterate - acceleration * iterate -
          secondaryStepSize * gradient;
    }

    i += effectiveBatchSize;
    currentFunction += effectiveBatchSize;
  }

  Info << "MaSS: maximum iterations (" << maxIterations << ") reached; "
      << "terminating optimization." << std::endl;

  // Calculate final objective.
  overallObjective = 0;
  for (size_t i = 0; i < numFunctions; i += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize, numFunctions - i);
    overallObjective += f.Evaluate(iterate, i, effectiveBatchSize);
  }
  return overallObjective;
}

} // namespace ens

#endif
