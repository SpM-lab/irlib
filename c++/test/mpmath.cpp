#include "gtest.h"

#include <fstream>

#include <Eigen/MPRealSupport>
#include <Eigen/SVD>
//#include <Eigen/LU>

//#include "../include/irlib/gauss_legendre.hpp"
#include <irlib/gauss_legendre.hpp>

using mpfr::mpreal;

TEST(mpmath, SVD) {

    mpreal::set_default_prec(167);

    // Declare matrix and vector types with multi-precision scalar type
    typedef Eigen::Matrix<mpreal,Eigen::Dynamic,Eigen::Dynamic>  MatrixXmp;
    typedef Eigen::Matrix<mpreal,Eigen::Dynamic,1>        VectorXmp;

    int N = 10;
    MatrixXmp A = MatrixXmp::Random(N,N);
    VectorXmp b = VectorXmp::Random(N);

    Eigen::BDCSVD<MatrixXmp> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    ASSERT_TRUE(svd.matrixU().rows() == N);
    ASSERT_TRUE(svd.matrixU().cols() == N);
    ASSERT_TRUE(svd.matrixV().rows() == N);
    ASSERT_TRUE(svd.matrixV().cols() == N);
    MatrixXmp A_reconst = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().adjoint();

    ASSERT_TRUE((A - A_reconst).cwiseAbs().maxCoeff() < 1e-40);
}

TEST(mpmath, gauss_legenre) {
    mpreal::set_default_prec(167);

    //Integrate x**2 over [-1, 1]
    for (int degree=2; degree<=6; ++degree) {
        std::vector<std::pair<mpreal,mpreal>> nodes = gauss_legendre_nodes<mpreal>(degree);

        mpreal sum = 0.0;
        for (auto n : nodes) {
            sum += (n.first*n.first) * n.second;
        }
        ASSERT_TRUE(mpfr::abs(sum - mpreal(2.0)/mpreal(3.0)) < 1e-48);
    }
}
