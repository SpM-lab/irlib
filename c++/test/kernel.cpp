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

using namespace irlib;

TEST(kernel, composite_gauss_legendre_integration) {
    int num_sec = 10;
    int num_local_nodes = 12;

    IR_MPREAL::set_default_prec(167);

    auto local_nodes = detail::gauss_legendre_nodes<IR_MPREAL>(num_local_nodes);
    auto section_edges_x = linspace<IR_MPREAL>(-1, 1, num_sec+1);
    auto global_nodes = composite_gauss_legendre_nodes(section_edges_x, local_nodes);

    IR_MPREAL sum = 0.0;
    for (auto n : global_nodes) {
        sum += (n.first*n.first) * n.second;
    }

    ASSERT_TRUE(abs(sum - IR_MPREAL(2.0)/IR_MPREAL(3.0)) < 1e-48);

}

TEST(kernel, matrixrep) {
    typedef Eigen::Matrix<IR_MPREAL,Eigen::Dynamic,Eigen::Dynamic>  MatrixXmp;

    int deci_deg = 50;
    IR_MPREAL::set_default_prec(int(deci_deg*3.333));

    int num_sec = 1;
    int Nl = 6;
    int gauss_legendre_deg = 12;

    std::vector<IR_MPREAL> section_edges_x = linspace<IR_MPREAL>(-1, 1, num_sec+1);
    std::vector<IR_MPREAL> section_edges_y = linspace<IR_MPREAL>(-1, 1, num_sec+1);

    auto const_kernel = [](const IR_MPREAL& x, const IR_MPREAL& y) {return 1.0;};
    auto Kmat = matrix_rep(const_kernel, section_edges_x, section_edges_y, gauss_legendre_deg, Nl);

    ASSERT_TRUE(abs(Kmat(0,0) - IR_MPREAL(2)) < 1e-30);
    ASSERT_TRUE(abs(Kmat(1,1)) < 1e-30);
}

TEST(kernel, SVD) {
    typedef Eigen::Matrix<IR_MPREAL,Eigen::Dynamic,Eigen::Dynamic>  MatrixXmp;

    int deci_deg = 30;
    IR_MPREAL::set_default_prec(int(deci_deg*3.333));

    int num_sec = 20;
    int Nl = 6;
    double Lambda = 100.0;
    std::vector<int> num_local_nodes_list{12};
    for (int num_local_nodes : num_local_nodes_list) {
        std::vector<IR_MPREAL> section_edges_x = linspace<IR_MPREAL>(-1, 1, num_sec+1);
        std::vector<IR_MPREAL> section_edges_y = linspace<IR_MPREAL>(-1, 1, num_sec+1);

        fermionic_kernel<mpreal> kernel(Lambda);
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
        std::vector<IR_MPREAL> section_edges_x = linspace<IR_MPREAL>(0, 1, num_sec/2+1);
        std::vector<IR_MPREAL> section_edges_y = linspace<IR_MPREAL>(0, 1, num_sec/2+1);

        fermionic_kernel<mpreal> kernel(Lambda);
        auto kernel_even = [&](const IR_MPREAL& x, const IR_MPREAL& y) {
            return kernel(x, y) + kernel(x, -y);
        };
        auto kernel_odd = [&](const IR_MPREAL& x, const IR_MPREAL& y) {
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

TEST(kernel, transformation_to_matsubara) {
    int ns = 1000;
    int k = 3;

    Eigen::MatrixXd coeff(ns, k+1);
    coeff.setZero();
    for (int s=0; s<ns; ++s) {
        coeff(s,0) = std::sqrt(0.5);
    }

    auto section_edges = linspace<IR_MPREAL>(IR_MPREAL(0.0), IR_MPREAL(1.0), ns+1);

    auto u_basis = std::vector<pp_type>{pp_type(ns, section_edges, coeff)};

    auto n = 10000000000;
    std::vector<long> n_vec{n};
    Eigen::Tensor<std::complex<double>,2> Tnl;
    compute_transformation_matrix_to_matsubara<double>(n_vec, statistics::FERMIONIC, u_basis, Tnl);

    auto jump = 2*u_basis[0].compute_value(1.0);
    auto ref = jump * std::sqrt(2.0)/((2*n+1)*M_PI);
    ASSERT_NEAR(Tnl(0,0).imag()/ref, 1.0, 1e-8);
}

TEST(kernel, basis_functions) {
    IR_MPREAL::set_default_prec(169);

    double Lambda = 100.0;
    int max_dim = 30;
    int Nl = 10;

    fermionic_kernel<mpreal> kernel(Lambda);

    std::vector<double> sv;
    std::vector<pp_type> u_basis, v_basis;
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
        //ASSERT_TRUE(std::abs(u_basis[l].compute_value(x) - parity * u_basis[l].compute_value(-x)) < 1e-10);
        //ASSERT_TRUE(std::abs(v_basis[l].compute_value(x) - parity * v_basis[l].compute_value(-x)) < 1e-10);
        ASSERT_NEAR(u_basis[l].overlap(u_basis[l]), 0.5, 1e-10);
        ASSERT_NEAR(v_basis[l].overlap(v_basis[l]), 0.5, 1e-10);
        parity *= -1;
    }

    for (int l=0; l<sv.size()-2; l+=2) {
        ASSERT_NEAR(u_basis[l].overlap(u_basis[l+2]), 0.0, 1e-10);
        ASSERT_NEAR(v_basis[l].overlap(v_basis[l+2]), 0.0, 1e-10);
    }

    // l=16 and x=0.5
    ASSERT_NEAR(u_basis[16].compute_value(0.5)/u_basis[16].compute_value(1.0), 0.129752287857, 1e-8);
    ASSERT_NEAR(v_basis[16].compute_value(0.5)/v_basis[16].compute_value(1.0), 0.0619868246037, 1e-8);

    // l=19 and x=0.5
    ASSERT_NEAR(u_basis[19].compute_value(0.5)/u_basis[19].compute_value(1.0), -0.105239562038, 1e-8);
    ASSERT_NEAR(v_basis[19].compute_value(0.5)/v_basis[19].compute_value(1.0), -0.378649485397, 1e-8);

    auto n = 10000000000;
    std::vector<long> n_vec{n};
    Eigen::Tensor<std::complex<double>,2> Tnl;
    compute_transformation_matrix_to_matsubara<double>(n_vec, statistics::FERMIONIC, u_basis, Tnl);

    for (int l=0; l<u_basis.size(); ++l) {
        if (l%2==0) {
            auto jump = 2*u_basis[l].compute_value(1.0);
            auto ref = jump * std::sqrt(2.0)/((2*n+1)*M_PI);
            ASSERT_NEAR(Tnl(0,l).imag()/ref, 1.0, 1e-8);
        } else {
            ASSERT_NEAR(Tnl(0,l).imag(), 0.0, 1e-10);
        }
    }
}


TEST(kernel, Ik) {
    double x0 = 0.99;
    double x1 = 1.00;
    double dx = x1 - x0;

    double w = 100.0;

    std::complex<double> z = std::complex<double>(0.0, w);

    std::complex<double> dx_z = dx * z;
    std::complex<double> dx_z2 = dx_z * dx_z;
    std::complex<double> dx_z3 = dx_z2 * dx_z;
    std::complex<double> inv_z = 1.0 / z;
    std::complex<double> inv_z2 = inv_z * inv_z;
    std::complex<double> inv_z3 = inv_z2 * inv_z;
    std::complex<double> inv_z4 = inv_z3 * inv_z;
    std::complex<double> exp = std::exp(dx * z);
    std::complex<double> exp0 = std::exp((static_cast<double>(x0) + 1.0) * z);

    auto I0 = (-1.0 + exp) * inv_z * exp0;
    auto I1 = ((dx_z - 1.0) * exp + 1.0) * inv_z2 * exp0;
    auto I2 = ((dx_z2 - 2.0 * dx_z + 2.0) * exp - 2.0) * inv_z3 * exp0;
    auto I3 = ((dx_z3 - 3.0 * dx_z2 + 6.0 * dx_z - 6.0) * exp + 6.0) * inv_z4 * exp0;

    int K = 3;
    std::vector<std::complex<double> > Ik(K+1);
    compute_Ik(x0, dx, w, K, Ik);

    ASSERT_TRUE(std::abs((I0-Ik[0])/I0) < 1e-8);
    ASSERT_TRUE(std::abs((I1-Ik[1])/I1) < 1e-8);
    ASSERT_TRUE(std::abs((I2-Ik[2])/I2) < 1e-8);
    ASSERT_TRUE(std::abs((I3-Ik[3])/I3) < 1e-8);
}
