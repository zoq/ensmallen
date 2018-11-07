/**
 * @file ma_ss_test.cpp
 * @author Marcus Edel
 *
 * Test file for the MaSS optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace ens;
using namespace ens::test;

/**
 * Run EVE on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("MaSSLogisticRegressionTest","[MaSSTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  // EVE optimizer(1e-3, 1, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  MaSS optimizer(0.001, 0.013, 1, 0.001, 500000, 1e-8, true);
  arma::mat coordinates = lr.GetInitialPoint();
  std::cout << coordinates << std::endl;
  optimizer.Optimize(lr, coordinates);
  std::cout << coordinates << std::endl;

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Test the MaSS optimizer on the Sphere function.
 */
TEST_CASE("MaSSSphereFunctionTest","[MaSSTest]")
{
  SphereFunction f(2);
  MaSS optimizer(0.001, 0.013, 2, 0.001, 500000, 1e-8, true);

  arma::mat coordinates = f.GetInitialPoint();

  std::cout << coordinates << std::endl;
  optimizer.Optimize(f, coordinates);
  std::cout << coordinates << std::endl;

  REQUIRE(coordinates[0] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[1] == Approx(0.0).margin(0.1));
}

/**
 * Test the MaSS optimizer on the Styblinski-Tang function.
 */
TEST_CASE("MaSSStyblinskiTangFunctionTest","[MaSSTest]")
{
  StyblinskiTangFunction f(2);
  MaSS optimizer(0.001, 0.013, 2, 0.001, 500000, 1e-8, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(-2.9).epsilon(0.01));
  REQUIRE(coordinates[1] == Approx(-2.9).epsilon(0.01));
}
