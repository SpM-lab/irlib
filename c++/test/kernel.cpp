#include "gtest.h"

#include <irlib/kernel.hpp>
#include <irlib/detail/aux.hpp>
#include <irlib/detail/gauss_legendre.hpp>

#include <fstream>
#include <vector>

#include <Eigen/MPRealSupport>
#include <Eigen/SVD>
//#include <Eigen/LU>

//#include "../include/irlib/gauss_legendre.hpp"
#include <irlib/detail/gauss_legendre.hpp>
#include <irlib/detail/legendre_polynomials.hpp>

using mpfr::mpreal;


TEST(kernel, composite_gauss_legendre_integration) {
    int num_sec = 10;
    int num_local_nodes = 12;

    mpreal::set_default_prec(167);

    auto local_nodes = irlib::detail::gauss_legendre_nodes<mpreal>(num_local_nodes);
    auto section_edges_x = irlib::linspace<mpreal>(-1, 1, num_sec+1);
    auto global_nodes = irlib::detail::composite_gauss_legendre_nodes(section_edges_x, local_nodes);

    mpreal sum = 0.0;
    for (auto n : global_nodes) {
        sum += (n.first*n.first) * n.second;
    }

    ASSERT_TRUE(mpfr::abs(sum - mpreal(2.0)/mpreal(3.0)) < 1e-48);

}

TEST(kernel, matrixrep) {
    typedef Eigen::Matrix<mpreal,Eigen::Dynamic,Eigen::Dynamic>  MatrixXmp;

    int deci_deg = 50;
    mpreal::set_default_prec(int(deci_deg*3.333));

    int num_sec = 1;
    int Nl = 6;
    int gauss_legendre_deg = 12;

    std::vector<mpreal> section_edges_x = irlib::linspace<mpreal>(-1, 1, num_sec+1);
    std::vector<mpreal> section_edges_y = irlib::linspace<mpreal>(-1, 1, num_sec+1);

    auto const_kernel = [](const mpreal& x, const mpreal& y) {return 1.0;};
    auto Kmat = irlib::matrix_rep(const_kernel, section_edges_x, section_edges_y, gauss_legendre_deg, Nl);

    ASSERT_TRUE(mpfr::abs(Kmat(0,0) - mpreal(2)) < 1e-30);
    ASSERT_TRUE(mpfr::abs(Kmat(1,1)) < 1e-30);
}

TEST(kernel, SVD) {
    typedef Eigen::Matrix<mpreal,Eigen::Dynamic,Eigen::Dynamic>  MatrixXmp;

    int deci_deg = 30;
    mpreal::set_default_prec(int(deci_deg*3.333));

    int num_sec = 20;
    int Nl = 6;
    double Lambda = 100.0;
    std::vector<int> num_local_nodes_list{12};
    for (int num_local_nodes : num_local_nodes_list) {
        std::vector<mpreal> section_edges_x = irlib::linspace<mpreal>(-1, 1, num_sec+1);
        std::vector<mpreal> section_edges_y = irlib::linspace<mpreal>(-1, 1, num_sec+1);

        irlib::fermionic_kernel kernel(Lambda);
        auto Kmat = irlib::matrix_rep(kernel, section_edges_x, section_edges_y, num_local_nodes, Nl);

        Eigen::BDCSVD<MatrixXmp> svd(Kmat, Eigen::ComputeFullU | Eigen::ComputeFullV);

        {
            int l = 1;
            auto tmp = svd.singularValues()(l)/svd.singularValues()(0);
            ASSERT_TRUE(mpfr::abs((tmp  - 0.853837813)/tmp) < 1e-5);
        }
    }

    //half interval [0, 1]
    for (int num_local_nodes : num_local_nodes_list) {
        std::vector<mpreal> section_edges_x = irlib::linspace<mpreal>(0, 1, num_sec/2+1);
        std::vector<mpreal> section_edges_y = irlib::linspace<mpreal>(0, 1, num_sec/2+1);

        irlib::fermionic_kernel kernel(Lambda);
        auto kernel_even = [&](const mpreal& x, const mpreal& y) {
            return kernel(x, y) + kernel(x, -y);
        };
        auto kernel_odd = [&](const mpreal& x, const mpreal& y) {
            return kernel(x, y) - kernel(x, -y);
        };
        auto Kmat_even = irlib::matrix_rep(kernel_even, section_edges_x, section_edges_y, num_local_nodes, Nl);
        auto Kmat_odd = irlib::matrix_rep(kernel_odd, section_edges_x, section_edges_y, num_local_nodes, Nl);

        Eigen::BDCSVD<MatrixXmp> svd_even(Kmat_even, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::BDCSVD<MatrixXmp> svd_odd(Kmat_odd, Eigen::ComputeFullU | Eigen::ComputeFullV);

        {
            auto tmp = svd_odd.singularValues()(0)/svd_even.singularValues()(0);
            ASSERT_TRUE(mpfr::abs((tmp  - 0.853837813)/tmp) < 1e-5);
        }
    }

}

