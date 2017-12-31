#include "gtest.h"

#include <fstream>
#include <utility>
#include <iomanip>

#include <Eigen/MPRealSupport>
#include <Eigen/SVD>
#include <irlib/common.hpp>
#include <irlib/detail/gauss_legendre.hpp>
#include <irlib/detail/legendre_polynomials.hpp>

using namespace irlib;

TEST(mpmath, cast) {
    ir_set_default_prec<mpreal>(167);

    for (int i=0; i<100; ++i) {
        ASSERT_TRUE(mpreal(i) - mpreal(std::to_string(i)) == 0);
    }

    ASSERT_TRUE(mpreal(0.5) - mpreal("0.5") == 0);

}

TEST(mpmath, SVD) {

    ir_set_default_prec<mpreal>(167);

    //mpreal x{1};
    //test<mpreal>(x, x);
    //make_pair_test<mpreal,mpreal>(x, x);

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
    ir_set_default_prec<mpreal>(167);

    //Integrate x**2 over [-1, 1]
    for (int degree : std::vector<int>{6, 12, 24}) {
        std::vector<std::pair<mpreal,mpreal>> nodes = irlib::detail::gauss_legendre_nodes<mpreal>(degree);

        mpreal sum = 0.0;
        for (auto n : nodes) {
            sum += (n.first*n.first) * n.second;
        }
        ASSERT_TRUE(abs(sum - mpreal(2.0)/mpreal(3.0)) < 1e-48);
    }
}

TEST(mpmath, legendre_polynomials) {
    ir_set_default_prec<mpreal>(167);

    int Nl = 100;

    mpreal x("0.5");

    std::vector<mpreal> vals(Nl);

    for (int l=0; l<Nl; ++l) {
        vals[l] = irlib::legendre_p(l, x);
    }

    for (int l=1; l<Nl-1; ++l) {
        auto mp_l(l);
        auto left_side = (mp_l+1) * vals[l+1];
        auto right_side = (2*mp_l+1) * x * vals[l] - mp_l * vals[l-1];
        ASSERT_TRUE(abs(left_side-right_side) < 1e-40);
    }
}


TEST(mpmath, normalized_legendre_polynomials_derivatives) {
    ir_set_default_prec<mpreal>(167);

    int Nl = 3;
    mpreal x(1);
    auto deriv = irlib::normalized_legendre_p_derivatives(Nl, x);

    //0-th normalized Legendre polynomial
    ASSERT_TRUE(abs(deriv[0][0]-1/sqrt(2)) < 1e-40);
    ASSERT_TRUE(abs(deriv[0][1]) < 1e-40);

    //1-th normalized Legendre polynomial
    ASSERT_TRUE(abs(deriv[1][0]-sqrt(mpreal(3)/mpreal(2))) < 1e-40);
    ASSERT_TRUE(abs(deriv[1][1]-sqrt(mpreal(3)/mpreal(2))) < 1e-40);

    //2-th normalized Legendre polynomial
    auto f0 = sqrt(mpreal(5)/mpreal(2));
    auto f1 = sqrt(mpreal(5)/mpreal(2)) * mpreal(3);
    ASSERT_TRUE(abs(deriv[2][0]-f0) < 1e-40);
    ASSERT_TRUE(abs(deriv[2][1]-f1) < 1e-40);
    ASSERT_TRUE(abs(deriv[2][2]-f1) < 1e-40);

}

/*
TEST(mpmath, io) {
    ir_set_default_prec<mpreal>(200);

    mpreal mp_tmp = mpreal("0.000001")/mpreal("3");
    ir_set_default_prec<mpreal>(20);

    {
        std::ofstream ofs("mp_math_io.txt");
        ofs << std::setprecision(mpfr::bits2digits(mp_tmp.get_prec())) << mp_tmp << std::endl;
        std::cout << std::setprecision(mpfr::bits2digits(mp_tmp.get_prec())) << mp_tmp << std::endl;
    }

    {
        mpreal mp_tmp2;
        std::ifstream ifs("mp_math_io.txt");
        ir_set_default_prec<mpreal>(20);
        ifs >> mp_tmp2;


        mpreal mp_tmp3 = mpreal("0.000001")/mpreal("3");
        mpreal diff = mp_tmp - mp_tmp2;
        std::cout << "diff " << mp_tmp.get_prec() << " " << mp_tmp2.get_prec() << " " << diff.get_prec() << " " << mp_tmp3.get_prec() << std::endl;
        std::cout << std::setprecision(mpfr::bits2digits(mp_tmp.get_prec())) << mp_tmp - mp_tmp2 << std::endl;
    }

}
*/
