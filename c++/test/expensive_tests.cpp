#include "common.hpp"

#include <fstream>

using namespace irlib;

TEST(kernel, composite_gauss_legendre_integration) {
    int num_sec = 10;
    int num_local_nodes = 12;

    ir_set_default_prec<mpreal>(167);

    auto local_nodes = detail::gauss_legendre_nodes<mpreal>(num_local_nodes);
    auto section_edges_x = linspace<mpreal>(-1, 1, num_sec + 1);
    auto global_nodes = composite_gauss_legendre_nodes(section_edges_x, local_nodes);

    mpreal sum = 0.0;
    for (auto n : global_nodes) {
        sum += (n.first * n.first) * n.second;
    }

    ASSERT_TRUE(abs(sum - mpreal(2.0) / mpreal(3.0)) < 1e-48);

}

TEST(kernel, matrixrep) {
    typedef Eigen::Matrix<mpreal, Eigen::Dynamic, Eigen::Dynamic> MatrixXmp;

    int deci_deg = 50;
    ir_set_default_prec<mpreal>(ir_digits2bits(deci_deg));

    int num_sec = 1;
    int Nl = 6;
    int gauss_legendre_deg = 12;

    std::vector<mpreal> section_edges_x = linspace<mpreal>(-1, 1, num_sec + 1);
    std::vector<mpreal> section_edges_y = linspace<mpreal>(-1, 1, num_sec + 1);

    auto const_kernel = [](const mpreal &x, const mpreal &y) { return mpreal(1.0); };
    auto Kmat = matrix_rep<mpreal>(const_kernel, section_edges_x, section_edges_y, gauss_legendre_deg, Nl);

    ASSERT_TRUE(abs(Kmat(0, 0) - mpreal(2)) < 1e-30);
    ASSERT_TRUE(abs(Kmat(1, 1)) < 1e-30);
}

TEST(kernel, SVD) {
    typedef Eigen::Matrix<mpreal, Eigen::Dynamic, Eigen::Dynamic> MatrixXmp;

    int deci_deg = 30;
    ir_set_default_prec<mpreal>(ir_digits2bits(deci_deg));

    int num_sec = 20;
    int Nl = 6;
    double Lambda = 100.0;
    std::vector<int> num_local_nodes_list{12};
    for (int num_local_nodes : num_local_nodes_list) {
        std::vector<mpreal> section_edges_x = linspace<mpreal>(-1, 1, num_sec + 1);
        std::vector<mpreal> section_edges_y = linspace<mpreal>(-1, 1, num_sec + 1);

        fermionic_kernel<mpreal> kernel(Lambda);
        auto Kmat = matrix_rep<mpreal>(kernel, section_edges_x, section_edges_y, num_local_nodes, Nl);

        Eigen::BDCSVD<MatrixXmp> svd(Kmat, Eigen::ComputeFullU | Eigen::ComputeFullV);

        {
            int l = 1;
            auto tmp = svd.singularValues()(l) / svd.singularValues()(0);
            ASSERT_TRUE(abs((tmp - 0.853837813) / tmp) < 1e-5);
        }
    }

    //half interval [0, 1]
    for (int num_local_nodes : num_local_nodes_list) {
        std::vector<mpreal> section_edges_x = linspace<mpreal>(0, 1, num_sec / 2 + 1);
        std::vector<mpreal> section_edges_y = linspace<mpreal>(0, 1, num_sec / 2 + 1);

        fermionic_kernel<mpreal> kernel(Lambda);
        auto kernel_even = [&](const mpreal &x, const mpreal &y) {
            return kernel(x, y) + kernel(x, -y);
        };
        auto kernel_odd = [&](const mpreal &x, const mpreal &y) {
            return kernel(x, y) - kernel(x, -y);
        };
        auto Kmat_even = matrix_rep<mpreal>(kernel_even, section_edges_x, section_edges_y, num_local_nodes, Nl);
        auto Kmat_odd = matrix_rep<mpreal>(kernel_odd, section_edges_x, section_edges_y, num_local_nodes, Nl);

        Eigen::BDCSVD<MatrixXmp> svd_even(Kmat_even, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::BDCSVD<MatrixXmp> svd_odd(Kmat_odd, Eigen::ComputeFullU | Eigen::ComputeFullV);

        ASSERT_TRUE(abs(svd_even.singularValues()[0] - 0.205608636) < 1e-5);
        ASSERT_TRUE(abs(svd_odd.singularValues()[0] - 0.175556428) < 1e-5);
        {
            auto tmp = svd_odd.singularValues()(0) / svd_even.singularValues()(0);
            ASSERT_TRUE(abs((tmp - 0.853837813) / tmp) < 1e-5);
        }
    }

}

TEST(kernel, transformation_to_matsubara) {
    int ns = 1000;
    int k = 4;

    MatrixXmp coeff(ns, k + 1);
    coeff.setZero();
    for (int s = 0; s < ns; ++s) {
        coeff(s, 0) = mpfr::sqrt(mpreal("0.5"));
    }

    auto section_edges = linspace<mpreal>(mpreal(0.0), mpreal(1.0), ns + 1);

    auto u_basis = std::vector<pp_type>{pp_type(ns, section_edges, coeff)};

    auto n = 10000000000;
    std::vector<long> n_vec{n};
    Eigen::Tensor<std::complex<double>, 2> Tnl;
    compute_transformation_matrix_to_matsubara<mpfr::mpreal>(n_vec, statistics::FERMIONIC, u_basis, Tnl);

    auto jump = 2 * u_basis[0].compute_value(1.0);
    auto ref = static_cast<double>(jump * std::sqrt(2.0) / ((2 * n + 1) * M_PI));
    ASSERT_NEAR(Tnl(0, 0).imag() / ref, 1.0, 1e-8);
}

TEST(basis, Lambda100) {
    double Lambda = 100.0;

    basis b = compute_basis(statistics::FERMIONIC, Lambda);
    savetxt("b_io.txt", b);
    basis b2 = loadtxt("b_io.txt");

    ASSERT_TRUE(b.dim() == b2.dim());

    double x = 0.99999;
    double y = 0.00001;
    double eps = 1e-10;
    for (int l = 0; l < b.dim(); ++l) {
        ASSERT_NEAR(b.sl(l), b2.sl(l), eps);
        ASSERT_NEAR(b.ulx(l, x), b2.ulx(l, x), eps);
        ASSERT_NEAR(b.vly(l, y), b2.vly(l, y), eps);
    }
}

TEST(kernel, basis_functions) {
    ir_set_default_prec<mpreal>(169);

    double Lambda = 100.0;
    int max_dim = 30;

    fermionic_kernel<mpreal> kernel(Lambda);

    std::vector<mpreal> sv;
    std::vector<pp_type> u_basis, v_basis;
    std::vector<std::vector<mpreal>> u_basis_coeff_l;
    std::vector<std::vector<mpreal>> v_basis_coeff_l;
    std::tie(sv, u_basis, v_basis, u_basis_coeff_l, v_basis_coeff_l) = generate_ir_basis_functions<mpreal>(kernel, max_dim, 1e-12);

    // Check singular values
    ASSERT_NEAR(static_cast<double>(sv[0]), 0.205608636, 1e-8);
    ASSERT_NEAR(static_cast<double>(sv[1]), 0.175556428, 1e-8);
    ASSERT_NEAR(static_cast<double>(sv[18]), 2.30686654e-06, 1e-13);
    ASSERT_NEAR(static_cast<double>(sv[28]), 1.42975303e-10, 1e-18);
    ASSERT_NEAR(static_cast<double>(sv[29]), 4.85605434e-11, 1e-18);

    double x = 0.111111;
    int parity = 1;
    for (int l = 0; l < sv.size(); ++l) {
        ASSERT_NEAR(static_cast<double>(u_basis[l].overlap(u_basis[l])), 0.5, 1e-10);
        ASSERT_NEAR(static_cast<double>(v_basis[l].overlap(v_basis[l])), 0.5, 1e-10);
        parity *= -1;
    }

    for (int l = 0; l < sv.size() - 2; l += 2) {
        ASSERT_NEAR(static_cast<double>(u_basis[l].overlap(u_basis[l + 2])), 0.0, 1e-10);
        ASSERT_NEAR(static_cast<double>(v_basis[l].overlap(v_basis[l + 2])), 0.0, 1e-10);
    }

    // l=16 and x=0.5
    ASSERT_NEAR(static_cast<double>(u_basis[16].compute_value(0.5) / u_basis[16].compute_value(1.0)), 0.129752287857, 1e-8);
    ASSERT_NEAR(static_cast<double>(v_basis[16].compute_value(0.5) / v_basis[16].compute_value(1.0)), 0.0619868246037, 1e-8);

    // l=19 and x=0.5
    ASSERT_NEAR(static_cast<double>(u_basis[19].compute_value(0.5) / u_basis[19].compute_value(1.0)), -0.105239562038, 1e-8);
    ASSERT_NEAR(static_cast<double>(v_basis[19].compute_value(0.5) / v_basis[19].compute_value(1.0)), -0.378649485397, 1e-8);

    auto n = 10000000000;
    std::vector<long> n_vec{n};
    Eigen::Tensor<std::complex<double>, 2> Tnl;
    compute_transformation_matrix_to_matsubara<mpreal>(n_vec, statistics::FERMIONIC, u_basis, Tnl);

    for (int l = 0; l < u_basis.size(); ++l) {
        if (l % 2 == 0) {
            auto jump = 2 * u_basis[l].compute_value(1.0);
            auto ref = jump * std::sqrt(2.0) / ((2 * n + 1) * M_PI);
            ASSERT_NEAR(static_cast<double>(Tnl(0, l).imag() / ref), 1.0, 1e-8);
        } else {
            ASSERT_NEAR(static_cast<double>(Tnl(0, l).imag()), 0.0, 1e-10);
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
    std::vector<std::complex<double> > Ik(K + 1);
    compute_Ik(x0, dx, w, K, Ik);

    ASSERT_TRUE(std::abs((I0 - Ik[0]) / I0) < 1e-8);
    ASSERT_TRUE(std::abs((I1 - Ik[1]) / I1) < 1e-8);
    ASSERT_TRUE(std::abs((I2 - Ik[2]) / I2) < 1e-8);
    ASSERT_TRUE(std::abs((I3 - Ik[3]) / I3) < 1e-8);
}


TEST(PiecewisePolynomial, Orthogonalization) {
    typedef double Scalar;
    const int n_section = 10, k = 8, n_basis = 3;
    typedef irlib::piecewise_polynomial<Scalar, mpreal> pp_type;

    ir_set_default_prec<mpreal>(167);

    std::vector<mpreal> section_edges(n_section + 1);
    Eigen::Tensor<Scalar, 3> coeff(n_basis, n_section, k + 1);

    for (int s = 0; s < n_section + 1; ++s) {
        section_edges[s] = s * 2.0 / n_section - 1.0;
    }
    section_edges[0] = -1.0;
    section_edges[n_section] = 1.0;

    std::vector<pp_type> nfunctions;

    // x^0, x^1, x^2, ...
    for (int n = 0; n < n_basis; ++n) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> coeff(n_section, k + 1);
        coeff.setZero();

        for (int s = 0; s < n_section; ++s) {
            double rtmp = 1.0;
            for (int l = 0; l < k + 1; ++l) {
                if (n - l < 0) {
                    break;
                }
                if (l > 0) {
                    rtmp /= l;
                    rtmp *= n + 1 - l;
                }
                coeff(s, l) = static_cast<double>(rtmp * pow(section_edges[s], n - l));
            }
        }

        nfunctions.push_back(pp_type(n_section, section_edges, coeff));
    }

    // Check if correctly constructed
    double x = 0.9;
    for (int n = 0; n < n_basis; ++n) {
        EXPECT_NEAR(static_cast<double>(nfunctions[n].compute_value(x)), std::pow(x, n), 1e-8);
    }

    // Check overlap
    for (int n = 0; n < n_basis; ++n) {
        for (int m = 0; m < n_basis; ++m) {
            EXPECT_NEAR(static_cast<double>(nfunctions[n].overlap(nfunctions[m])),
                        (std::pow(1.0, n + m + 1) - std::pow(-1.0, n + m + 1)) / (n + m + 1), 1e-8);
        }
    }


    // Check plus and minus
    for (int n = 0; n < n_basis; ++n) {
        EXPECT_NEAR(static_cast<double>(4 * nfunctions[n].compute_value(x)), static_cast<double>((4.0 * nfunctions[n]).compute_value(x)), 1e-8);
        for (int m = 0; m < n_basis; ++m) {
            EXPECT_NEAR(static_cast<double>(nfunctions[n].compute_value(x) + nfunctions[m].compute_value(x)),
                        static_cast<double>((nfunctions[n] + nfunctions[m]).compute_value(x)), 1e-8);
            EXPECT_NEAR(static_cast<double>(nfunctions[n].compute_value(x) - nfunctions[m].compute_value(x)),
                        static_cast<double>((nfunctions[n] - nfunctions[m]).compute_value(x)), 1e-8);
        }
    }

    irlib::orthonormalize(nfunctions);
    for (int n = 0; n < n_basis; ++n) {
        for (int m = 0; m < n_basis; ++m) {
            EXPECT_NEAR(static_cast<double>(nfunctions[n].overlap(nfunctions[m])),
                        n == m ? 1.0 : 0.0,
                        1e-8
            );
        }
    }

    //l = 0 should be x
    EXPECT_NEAR(static_cast<double>(nfunctions[1].compute_value(x) * std::sqrt(2.0 / 3.0)), x, 1E-8);
}



TEST(computeTnl, NegativeFreq) {
    double Lambda = 10.0;

    {
        auto bb = compute_basis(statistics::BOSONIC, Lambda, 1000, 1e-8, "mp");
        auto Tnl = bb.compute_Tnl(std::vector<long>{-1, 1});
        auto Tnl2 = bb.compute_Tnl(std::vector<long>{1, -1});
        auto l = 0;
        for (int l = 0; l < bb.dim(); ++l) {
            ASSERT_TRUE(Tnl(0, l) == std::conj(Tnl(1, l)));

            ASSERT_TRUE(Tnl(0, l) == Tnl2(1, l));
            ASSERT_TRUE(Tnl(1, l) == Tnl2(0, l));
        }
    }
}

//template<class T>
//class HighTTest : public testing::Test {
//};
//typedef ::testing::Types<irlib::basis_f, irlib::basis_b> BasisTypes;
//TYPED_TEST_CASE(HighTTest, BasisTypes);

class TestAllStatistics : public ::testing::TestWithParam<statistics::statistics_type> {
};


TEST_P(TestAllStatistics, HighT) {
    try {
        //construct ir basis
        const double Lambda = 0.01;//high T
        const int max_dim = 100;
        const double cutoff = 1e-8;
        basis b = compute_basis(GetParam(), Lambda, max_dim, cutoff);
        ASSERT_TRUE(b.dim() >= 3);

        //IR basis functions should match Legendre polynomials
        const int N = 10;
        for (int i = 1; i < N - 1; ++i) {
            const double x = i * (2.0 / (N - 1)) - 1.0;

            double rtmp;

            //l = 0
            rtmp = b.ulx(0, x);
            ASSERT_TRUE(std::abs(rtmp - std::sqrt(0 + 0.5)) < 0.02);

            //l = 1
            rtmp = b.ulx(1, x);
            ASSERT_TRUE(std::abs(rtmp - std::sqrt(1 + 0.5) * x) < 0.02);

            //l = 2
            rtmp = b.ulx(2, x);
            ASSERT_TRUE(std::abs(rtmp - std::sqrt(2 + 0.5) * (1.5 * x * x - 0.5)) < 0.02);
        }

        //check parity
        {
            double sign = -1.0;
            double x = 1.0;
            for (int l = 0; l < b.dim(); ++l) {
                ASSERT_NEAR(b.ulx(l, x) + sign * b.ulx(l, -x), 0.0, 1e-8);
                sign *= -1;
            }
        }
    } catch (const std::exception &e) {
        FAIL() << e.what();
    }
}

INSTANTIATE_TEST_CASE_P(FermionBosonStatistics, TestAllStatistics,
                        ::testing::Values(statistics::statistics_type::FERMIONIC,
                                          statistics::statistics_type::BOSONIC));


/*
TEST(ComparisonMPvsDP, Fermion) {
    double Lambda = 1000.0;
    int max_dim = 10000;
    auto basis_mp = compute_basis(statistics::FERMIONIC, Lambda, 1000, 1e-6, "mp");
    auto basis_dp = compute_basis(statistics::FERMIONIC, Lambda, 1000, 1e-6, "long double");
    double tol = 1e-5;

    int Nl = std::min(basis_mp.dim(), basis_dp.dim());

    for (int s = 0; s < basis_mp.ul(0).num_sections(); ++s) {
        auto s0 = basis_dp.ul(0).section_edge(s);
        auto s1 = basis_dp.ul(0).section_edge(s + 1);
        auto xs = irlib::linspace<double>(static_cast<double>(s0), static_cast<double>(s1), 10);
        for (auto x : xs) {
            for (int l = 0; l < Nl; ++l) {
                ASSERT_NEAR(basis_dp.ulx(l, x), basis_mp.ulx(l, x), std::max(tol, tol * std::abs(basis_dp.ulx(l, x))));
            }
        }
    }
}
*/

template<class T>
class ExpansionByIRBasis : public testing::Test {
};

//typedef ::testing::Types<irlib::basis_f, irlib::basis_f_dp, irlib::basis_b, irlib::basis_b_dp> AllBasisTypes;
//TYPED_TEST_CASE(ExpansionByIRBasis, AllBasisTypes);

TEST(ExpansionByIRBasis, AllBasisTypes) {
    using accurate_fp_type = mpfr::mpreal;
    double cutoff = 1e-9;
    int max_dim = 1000;

    for (auto fp_mode : std::vector<std::string>{"mp"}) {
        for (auto statis : std::vector<statistics::statistics_type>{statistics::FERMIONIC,statistics::BOSONIC}) {
            for (auto beta : std::vector<double>{100.0}) {
                double Lambda = 3 * beta;
                int max_dim = 10000;
                double w_positive_pole = 0.6;
                double w_negative_pole = 0.4;

                auto b = compute_basis(statis, max_dim, Lambda, cutoff, fp_mode);

                ASSERT_TRUE(b.dim() > 0);

                double tol = 1e-7;

                typedef irlib::piecewise_polynomial<double, accurate_fp_type> pp_type;

                const int nptr = b.ul(0).num_sections() + 1;
                std::vector<mpreal> x(nptr);
                for (int i = 0; i < nptr; ++i) {
                    x[i] = b.ul(0).section_edge(i);
                }

                auto gx = [&](const accurate_fp_type &x) {
                    if (statis == statistics::FERMIONIC) {
                        if (-.5 * beta * x > 100.0) {
                            return w_positive_pole * exp(-0.5 * beta * (1 + x));
                        } else if (-.5 * beta * x < -100.0) {
                            return w_negative_pole * exp(-0.5 * beta * (1 - x));
                        } else {
                            //here we assume exp(0.5*beta) >> 1.
                            return exp(-0.5 * beta) *
                                   (w_positive_pole * exp(-0.5 * beta * x) + w_negative_pole * exp(0.5 * beta * x));
                        }
                    } else {
                        //do boson
                        if (-.5 * beta * x > 100.0) {
                            return w_positive_pole * exp(-0.5 * beta * (1 + x));
                        } else if (-.5 * beta * x < -100.0) {
                            return -w_negative_pole * exp(-0.5 * beta * (1 - x));
                        } else {
                            //here we assume exp(0.5*beta) >> 1.
                            return exp(-0.5 * beta) *
                                   (w_positive_pole * exp(-0.5 * beta * x) - w_negative_pole * exp(0.5 * beta * x));
                        }
                    }
                };

                //auto current_model{test_model<basis.get_statistics(), accurate_fp_type>()};
                //auto gx = [&](const accurate_fp_type & x) {
                //return current_model.gx(x, beta);
                //};

                std::vector<accurate_fp_type> section_edges;
                for (int s = 0; s < b.ul(0).num_sections() + 1; ++s) {
                    section_edges.push_back(-b.ul(0).section_edge(
                            b.ul(0).num_sections() - s
                    ));
                }
                for (int s = 0; s < b.ul(0).num_sections() + 1; ++s) {
                    section_edges.push_back(b.ul(0).section_edge(s));
                }

                std::vector<double> coeff(b.dim());
                for (int l = 0; l < b.dim(); ++l) {
                    auto f = [&](const accurate_fp_type &x) { return accurate_fp_type(gx(x) * b.ulx_mp(l, x)); };
                    coeff[l] = static_cast<double>(
                            irlib::integrate_gauss_legendre<accurate_fp_type, accurate_fp_type>(section_edges, f, 24) *
                            beta / std::sqrt(2.0));
                }

                std::vector<double> y_r(nptr, 0.0);
                for (int l = 0; l < b.dim(); ++l) {
                    for (int i = 0; i < nptr; ++i) {
                        y_r[i] += static_cast<double>(coeff[l] * (std::sqrt(2.0) / beta) * b.ulx_mp(l, x[i]));
                    }
                }

                double max_diff = 0.0;
                for (int i = 0; i < nptr; ++i) {
                    max_diff = std::max(
                            std::abs(static_cast<double>(gx(x[i]) - y_r[i])),
                            max_diff);
                }
                ASSERT_NEAR(max_diff, 0.0, tol);

                //to matsubara freq.
                std::vector<long> n_vec;
                for (int n = 0; n < 1000; ++n) {
                    n_vec.push_back(n);
                }
                //some higher frequencies
                n_vec.push_back(1000000);
                n_vec.push_back(100000000);
                int n_iw = n_vec.size();
                MatrixXc Tnl(n_iw, b.dim());
                auto Tnl_tensor = b.compute_Tnl(n_vec);
                MatrixXc coeff_vec(b.dim(), 1);
                for (int l = 0; l < b.dim(); ++l) {
                    coeff_vec(l, 0) = coeff[l];
                }
                MatrixXc coeff_iw = Eigen::Map<MatrixXc>(&Tnl_tensor(0, 0), n_vec.size(), b.dim()) * coeff_vec;

                const std::complex<double> zi(0.0, 1.0);
                int offset = (statis == irlib::statistics::FERMIONIC ? 1 : 0);
                for (int n = 0; n < n_iw; ++n) {
                    double wn = (2. * n_vec[n] + offset) * M_PI / beta;
                    std::complex<double> z = -w_positive_pole / (zi * wn - 1.0) - w_negative_pole / (zi * wn + 1.0);
                    ASSERT_NEAR(z.real(), coeff_iw(n).real(), tol);
                    ASSERT_NEAR(z.imag(), coeff_iw(n).imag(), tol);
                }
            }
        }
    }
}


