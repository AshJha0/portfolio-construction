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

TEST(Linalg, ReferenceParityReductionsAgreeOnExactData) {
  // On exactly-representable inputs every reduction order gives the same
  // value, so the parity kernels must agree with the plain ones there.
  const Vector a{1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0};
  const Vector b{0.5, 0.25, 0.125, 0.0625, 2.0, 4.0, 8.0, 16.0};
  EXPECT_DOUBLE_EQ(portopt::fma_dot(a, b), portopt::dot(a, b));
  EXPECT_DOUBLE_EQ(portopt::blas_dot(a, b), portopt::dot(a, b));
  EXPECT_DOUBLE_EQ(portopt::np_sum(a), 255.0);
  // np_sum spans all three regimes (naive < 8, unrolled <= 128, recursive).
  for (std::size_t n : {3u, 8u, 40u, 128u, 129u, 1250u}) {
    const Vector ones(n, 1.0);
    EXPECT_DOUBLE_EQ(portopt::np_sum(ones), static_cast<double>(n)) << n;
  }
  // blas_dot spans the block-splitting regimes (<= 384, (384, 768), >= 768).
  for (std::size_t n : {10u, 384u, 500u, 768u, 1250u}) {
    const Vector ones(n, 1.0);
    EXPECT_DOUBLE_EQ(portopt::blas_dot(ones, ones), static_cast<double>(n)) << n;
  }
  EXPECT_DOUBLE_EQ(portopt::kahan_sum(Vector(1250, 0.5)), 625.0);
}

TEST(Linalg, ParityMatVecKernelsMatchPlainProducts) {
  const Matrix m = testdata::sigma_ann();
  const Vector w(m.rows(), 0.125);
  const Vector plain = portopt::mat_vec(m, w);
  const Vector mv = portopt::np_matvec(m, w);
  // Symmetric matrix: the row and column products agree mathematically.
  const Vector vm = portopt::np_vecmat(w, m);
  ASSERT_EQ(mv.size(), plain.size());
  for (std::size_t i = 0; i < plain.size(); ++i) {
    EXPECT_NEAR(mv[i], plain[i], 1e-15) << i;
    EXPECT_NEAR(vm[i], plain[i], 1e-15) << i;
  }
  EXPECT_THROW(portopt::np_matvec(m, Vector{1.0, 2.0}), std::invalid_argument);
  EXPECT_THROW(portopt::np_vecmat(Vector{1.0, 2.0}, m), std::invalid_argument);
}

TEST(Linalg, CholSolveRejectsBadRhsLength) {
  const Matrix low = portopt::cholesky(testdata::spd3());
  EXPECT_THROW(portopt::chol_solve(low, Vector{1.0, 2.0}), std::invalid_argument);
}
