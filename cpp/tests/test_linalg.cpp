/// \file test_linalg.cpp
/// \brief Unit tests for the pinned Cholesky kernels and error paths.
#include <gtest/gtest.h>

#include <cmath>

#include "portopt/linalg.hpp"
#include "test_data.hpp"

using portopt::Matrix;
using portopt::Vector;

TEST(Linalg, CholeskyReconstructs) {
  const Matrix a = testdata::spd3();
  const Matrix low = portopt::cholesky(a);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      double acc = 0.0;
      for (std::size_t k = 0; k < 3; ++k) acc += low(i, k) * low(j, k);
      EXPECT_NEAR(acc, a(i, j), 1e-15) << i << "," << j;
      if (j > i) {
        EXPECT_EQ(low(i, j), 0.0);  // strictly lower triangular
      }
    }
  }
}

TEST(Linalg, SpdSolveResidual) {
  const Matrix a = testdata::spd3();
  const Vector b{1.0, -2.0, 0.5};
  const Vector x = portopt::spd_solve(a, b);
  const Vector ax = portopt::mat_vec(a, x);
  for (std::size_t i = 0; i < 3; ++i) EXPECT_NEAR(ax[i], b[i], 1e-12);
}

TEST(Linalg, SpdInverseIsIdentityProduct) {
  const Matrix a = testdata::spd3();
  const Matrix inv = portopt::spd_inverse(a);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      double acc = 0.0;
      for (std::size_t k = 0; k < 3; ++k) acc += a(i, k) * inv(k, j);
      EXPECT_NEAR(acc, i == j ? 1.0 : 0.0, 1e-12);
    }
  }
}

TEST(Linalg, DuplicatedAssetIsRejected) {
  // Singular covariance: third row/column duplicates the first.
  Matrix a(3, 3, 0.0);
  a(0, 0) = 0.04;  a(0, 1) = 0.01;  a(0, 2) = 0.04;
  a(1, 0) = 0.01;  a(1, 1) = 0.09;  a(1, 2) = 0.01;
  a(2, 0) = 0.04;  a(2, 1) = 0.01;  a(2, 2) = 0.04;
  EXPECT_THROW(portopt::cholesky(a), std::domain_error);
}

TEST(Linalg, IndefiniteMatrixIsRejected) {
  Matrix a(2, 2, 0.0);
  a(0, 0) = 1.0;  a(0, 1) = 2.0;
  a(1, 0) = 2.0;  a(1, 1) = 1.0;  // eigenvalues 3 and -1
  EXPECT_THROW(portopt::cholesky(a), std::domain_error);
}

TEST(Linalg, NonPositiveDiagonalIsRejected) {
  Matrix a(2, 2, 0.0);
  a(0, 0) = -1.0;
  a(1, 1) = 1.0;
  EXPECT_THROW(portopt::cholesky(a), std::domain_error);
}

TEST(Linalg, AsymmetricMatrixIsRejected) {
  Matrix a(2, 2, 0.0);
  a(0, 0) = 1.0;  a(0, 1) = 0.5;
  a(1, 0) = 0.4;  a(1, 1) = 1.0;
  EXPECT_THROW(portopt::check_square_symmetric(a), std::invalid_argument);
}

TEST(Linalg, NonSquareAndNaNAreRejected) {
  EXPECT_THROW(portopt::check_square_symmetric(Matrix(2, 3, 0.0)),
               std::invalid_argument);
  EXPECT_THROW(portopt::check_square_symmetric(Matrix(0, 0, 0.0)),
               std::invalid_argument);
  Matrix a(2, 2, 1.0);
  a(0, 1) = std::nan("");
  a(1, 0) = std::nan("");
  EXPECT_THROW(portopt::check_square_symmetric(a), std::invalid_argument);
}

TEST(Linalg, CompensatedSumIsAccurate) {
  // 1 + 100 * 1e-16 accumulated naively loses mass; Neumaier keeps it.
  Vector xs(101, 1e-16);
  xs[0] = 1.0;
  EXPECT_NEAR(portopt::kahan_sum(xs), 1.0 + 1e-14, 1e-18);
  EXPECT_DOUBLE_EQ(portopt::kahan_sum(Vector(1250, 0.5)), 625.0);
  EXPECT_DOUBLE_EQ(portopt::kahan_sum(Vector{}), 0.0);
}

TEST(Linalg, CholSolveRejectsBadRhsLength) {
  const Matrix low = portopt::cholesky(testdata::spd3());
  EXPECT_THROW(portopt::chol_solve(low, Vector{1.0, 2.0}), std::invalid_argument);
}
