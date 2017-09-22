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
using namespace irlib;


TEST(kernel, composite_gauss_legendre_integration) {
    int num_sec = 10;
    int num_local_nodes = 12;

    mpreal::set_default_prec(167);

    auto local_nodes = detail::gauss_legendre_nodes<mpreal>(num_local_nodes);
    auto section_edges_x = linspace<mpreal>(-1, 1, num_sec+1);
    auto global_nodes = detail::composite_gauss_legendre_nodes(section_edges_x, local_nodes);

    mpreal sum = 0.0;
    for (auto n : global_nodes) {
        sum += (n.first*n.first) * n.second;
    }

    ASSERT_TRUE(abs(sum - mpreal(2.0)/mpreal(3.0)) < 1e-48);

}

TEST(kernel, matrixrep) {
    typedef Eigen::Matrix<mpreal,Eigen::Dynamic,Eigen::Dynamic>  MatrixXmp;

    int deci_deg = 50;
    mpreal::set_default_prec(int(deci_deg*3.333));

    int num_sec = 1;
    int Nl = 6;
    int gauss_legendre_deg = 12;

    std::vector<mpreal> section_edges_x = linspace<mpreal>(-1, 1, num_sec+1);
    std::vector<mpreal> section_edges_y = linspace<mpreal>(-1, 1, num_sec+1);

    auto const_kernel = [](const mpreal& x, const mpreal& y) {return 1.0;};
    auto Kmat = matrix_rep(const_kernel, section_edges_x, section_edges_y, gauss_legendre_deg, Nl);

    ASSERT_TRUE(abs(Kmat(0,0) - mpreal(2)) < 1e-30);
    ASSERT_TRUE(abs(Kmat(1,1)) < 1e-30);
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
        std::vector<mpreal> section_edges_x = linspace<mpreal>(-1, 1, num_sec+1);
        std::vector<mpreal> section_edges_y = linspace<mpreal>(-1, 1, num_sec+1);

        fermionic_kernel kernel(Lambda);
        auto Kmat = matrix_rep(kernel, section_edges_x, section_edges_y, num_local_nodes, Nl);

        Eigen::BDCSVD<MatrixXmp> svd(Kmat, Eigen::ComputeFullU | Eigen::ComputeFullV);

        {
            int l = 1;
            auto tmp = svd.singularValues()(l)/svd.singularValues()(0);
            ASSERT_TRUE(abs((tmp  - 0.853837813)/tmp) < 1e-5);
        }
    }

    //half interval [0, 1]
    for (int num_local_nodes : num_local_nodes_list) {
        std::vector<mpreal> section_edges_x = linspace<mpreal>(0, 1, num_sec/2+1);
        std::vector<mpreal> section_edges_y = linspace<mpreal>(0, 1, num_sec/2+1);

        fermionic_kernel kernel(Lambda);
        auto kernel_even = [&](const mpreal& x, const mpreal& y) {
            return kernel(x, y) + kernel(x, -y);
        };
        auto kernel_odd = [&](const mpreal& x, const mpreal& y) {
            return kernel(x, y) - kernel(x, -y);
        };
        auto Kmat_even = matrix_rep(kernel_even, section_edges_x, section_edges_y, num_local_nodes, Nl);
        auto Kmat_odd = matrix_rep(kernel_odd, section_edges_x, section_edges_y, num_local_nodes, Nl);

        Eigen::BDCSVD<MatrixXmp> svd_even(Kmat_even, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::BDCSVD<MatrixXmp> svd_odd(Kmat_odd, Eigen::ComputeFullU | Eigen::ComputeFullV);

        ASSERT_TRUE(abs(svd_even.singularValues()[0] - 0.205608636) < 1e-5);
        ASSERT_TRUE(abs(svd_odd.singularValues()[0] - 0.175556428) < 1e-5);
        {
            auto tmp = svd_odd.singularValues()(0)/svd_even.singularValues()(0);
            ASSERT_TRUE(abs((tmp  - 0.853837813)/tmp) < 1e-5);
        }
    }

}


TEST(kernel, basis_functions) {
    mpreal::set_default_prec(169);

    double Lambda = 100.0;
    int max_dim = 30;
    int Nl = 10;

    fermionic_kernel kernel(Lambda);

    std::vector<double> sv;
    std::vector<piecewise_polynomial<double>> u_basis, v_basis;
    std::tie(sv,u_basis,v_basis) = generate_ir_basis_functions(kernel, max_dim, 1e-12, Nl);

    // Check singular values
    ASSERT_NEAR(sv[0],  0.205608636 ,  1e-8);
    ASSERT_NEAR(sv[1],  0.175556428 ,  1e-8);
    ASSERT_NEAR(sv[18],  2.30686654e-06 ,  1e-13);
    ASSERT_NEAR(sv[28],  1.42975303e-10 ,  1e-18);
    ASSERT_NEAR(sv[29],   4.85605434e-11 ,  1e-18);

    double x = 0.111111;
    int parity = 1;
    for (int l=0; l<sv.size(); ++l) {
        ASSERT_TRUE(std::abs(u_basis[l].compute_value(x) - parity * u_basis[l].compute_value(-x)) < 1e-10);
        ASSERT_TRUE(std::abs(v_basis[l].compute_value(x) - parity * v_basis[l].compute_value(-x)) < 1e-10);
        ASSERT_NEAR(u_basis[l].overlap(u_basis[l]), 1.0, 1e-10);
        ASSERT_NEAR(v_basis[l].overlap(v_basis[l]), 1.0, 1e-10);
        parity *= -1;
    }

    for (int l=0; l<sv.size()-1; ++l) {
        ASSERT_NEAR(u_basis[l].overlap(u_basis[l+1]), 0.0, 1e-10);
        ASSERT_NEAR(v_basis[l].overlap(v_basis[l+1]), 0.0, 1e-10);
    }

    // l=16 and x=0.5
    ASSERT_NEAR(u_basis[16].compute_value(0.5)/u_basis[16].compute_value(1.0), 0.129752287857, 1e-8);
    ASSERT_NEAR(v_basis[16].compute_value(0.5)/v_basis[16].compute_value(1.0), 0.0619868246037, 1e-8);

    // l=19 and x=0.5
    ASSERT_NEAR(u_basis[19].compute_value(0.5)/u_basis[19].compute_value(1.0), -0.105239562038, 1e-8);
    ASSERT_NEAR(v_basis[19].compute_value(0.5)/v_basis[19].compute_value(1.0), -0.378649485397, 1e-8);

}
