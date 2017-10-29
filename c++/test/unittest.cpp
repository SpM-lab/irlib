#include "common.hpp"

#include <fstream>

using namespace irlib;

TEST(PiecewisePolynomial, Orthogonalization) {
    typedef double Scalar;
    const int n_section = 10, k = 8, n_basis = 3;
    typedef irlib::piecewise_polynomial<Scalar,mpreal> pp_type;

    std::vector<mpreal> section_edges(n_section+1);
    Eigen::Tensor<Scalar,3> coeff(n_basis, n_section, k+1);

    for (int s = 0; s < n_section + 1; ++s) {
        section_edges[s] = s*2.0/n_section - 1.0;
    }
    section_edges[0] = -1.0;
    section_edges[n_section] = 1.0;

    std::vector<pp_type> nfunctions;

    // x^0, x^1, x^2, ...
    for (int n = 0; n < n_basis; ++ n) {
        Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> coeff(n_section, k+1);
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
                coeff(s, l) = static_cast<double>(rtmp * pow(section_edges[s], n-l));
            }
        }

        nfunctions.push_back(pp_type(n_section, section_edges, coeff));
    }

    // Check if correctly constructed
    double x = 0.9;
    for (int n = 0; n < n_basis; ++ n) {
        EXPECT_NEAR(nfunctions[n].compute_value(x), std::pow(x, n), 1e-8);
    }

    // Check overlap
    for (int n = 0; n < n_basis; ++ n) {
        for (int m = 0; m < n_basis; ++ m) {
            EXPECT_NEAR(nfunctions[n].overlap(nfunctions[m]), (std::pow(1.0,n+m+1)-std::pow(-1.0,n+m+1))/(n+m+1), 1e-8);
        }
    }


    // Check plus and minus
    for (int n = 0; n < n_basis; ++ n) {
        EXPECT_NEAR(4 * nfunctions[n].compute_value(x), (4.0*nfunctions[n]).compute_value(x), 1e-8);
        for (int m = 0; m < n_basis; ++m) {
            EXPECT_NEAR(nfunctions[n].compute_value(x) + nfunctions[m].compute_value(x),
                        (nfunctions[n] + nfunctions[m]).compute_value(x), 1e-8);
            EXPECT_NEAR(nfunctions[n].compute_value(x) - nfunctions[m].compute_value(x),
                        (nfunctions[n] - nfunctions[m]).compute_value(x), 1e-8);
        }
    }

    irlib::orthonormalize(nfunctions);
    for (int n = 0; n < n_basis; ++ n) {
        for (int m = 0; m < n_basis; ++m) {
            EXPECT_NEAR(nfunctions[n].overlap(nfunctions[m]),
                        n == m ? 1.0 : 0.0,
                        1e-8
            );
        }
    }

    //l = 0 should be x
    EXPECT_NEAR(nfunctions[1].compute_value(x) * std::sqrt(2.0/3.0), x, 1E-8);
}

template<class T>
class HighTTest : public testing::Test {
};

//typedef ::testing::Types<irlib::fermionic_kernel, irlib::bosonic_kernel> KernelTypes;
typedef ::testing::Types<irlib::basis_f, irlib::basis_b> BasisTypes;

TYPED_TEST_CASE(HighTTest, BasisTypes);

TYPED_TEST(HighTTest, BasisTypes) {
  try {
    //construct ir basis
    const double Lambda = 0.01;//high T
    const int max_dim = 100;
    TypeParam basis(Lambda, max_dim);
    ASSERT_TRUE(basis.dim()>3);

    //IR basis functions should match Legendre polynomials
    const int N = 10;
    for (int i = 1; i < N - 1; ++i) {
      const double x = i * (2.0/(N-1)) - 1.0;

      double rtmp;

      //l = 0
      rtmp = basis.ulx(0,x);
      ASSERT_TRUE(std::abs(rtmp-std::sqrt(0+0.5)) < 0.02);

      //l = 1
      rtmp = basis.ulx(1,x);
      ASSERT_TRUE(std::abs(rtmp-std::sqrt(1+0.5)*x) < 0.02);

      //l = 2
      rtmp = basis.ulx(2,x);
      ASSERT_TRUE(std::abs(rtmp-std::sqrt(2+0.5)*(1.5*x*x-0.5)) < 0.02);
    }

    //check parity
    {
      double sign = -1.0;
      double x = 1.0;
      for (int l = 0; l < basis.dim(); ++l) {
        ASSERT_NEAR(basis.ulx(l,x) + sign * basis.ulx(l,-x), 0.0, 1e-8);
        sign *= -1;
      }
    }
  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

TEST(ComparisonMPvsDP, Fermion) {
    double Lambda = 1000.0;
    int max_dim = 10000;
    irlib::basis_f basis_mp(Lambda, max_dim, 1e-6);
    irlib::basis_f_dp basis_dp(Lambda, max_dim, 1e-6);
    double tol = 1e-5;

    int Nl = std::min(basis_mp.dim(), basis_dp.dim());

    for (int s=0; s<basis_mp.ul(0).num_sections(); ++s) {
        auto s0 = basis_dp.ul(0).section_edge(s);
        auto s1 = basis_dp.ul(0).section_edge(s+1);
        auto xs = irlib::linspace<double>(static_cast<double>(s0), static_cast<double>(s1), 10);
        for (auto x : xs) {
            for (int l=0; l<Nl; ++l) {
                ASSERT_NEAR(basis_dp.ulx(l,x), basis_mp.ulx(l,x), std::max(tol, tol * std::abs(basis_dp.ulx(l,x))));
            }
        }
    }
}

template<class T>
class ExpansionByFermionBasis : public testing::Test {
};

//typedef ::testing::Types<irlib::basis_f> FermionBasisTypes;
//typedef ::testing::Types<irlib::basis_f, irlib::basis_f_dp> FermionBasisTypes;
typedef ::testing::Types<irlib::basis_f_dp> FermionBasisTypes;

TYPED_TEST_CASE(ExpansionByFermionBasis, FermionBasisTypes);

TYPED_TEST(ExpansionByFermionBasis, FermionBasisTypes) {
  using accurate_fp_type = typename TypeParam::accurate_fp_type;

  for (auto beta : std::vector<double>{100.0, 10000.0}) {
    double Lambda = 3*beta;
    int max_dim = 10000;
      //std::cout << "constructing " << std::endl;
    TypeParam  basis(Lambda, max_dim, 1e-8);
      //std::cout << "done " << std::endl;
    ASSERT_TRUE(basis.dim()>0);
      //std::cout << " beta " << beta << " " << basis.dim() << std::endl;

    //double tol = 1000*basis.sl(basis.dim()-1)/basis.sl(0);
    double tol = 1e-5;

    typedef irlib::piecewise_polynomial<double,accurate_fp_type> pp_type;

    const int nptr = basis.ul(0).num_sections() + 1;
    std::vector<mpreal> x(nptr);
    for (int i = 0; i < nptr; ++i) {
      x[i] = basis.ul(0).section_edge(i);
    }

    auto gtau = [&](const mpreal& x) {
        if (-.5*beta*x > 100.0) {
            return 0.5 * exp(-0.5*beta*(1+x));
        } else if (-.5*beta*x < -100.0) {
            return 0.5 * exp(-0.5*beta*(1-x));
        } else {
            return exp(-0.5*beta)*cosh(-0.5*beta*x);
        }

        //return exp(-0.5*scalar_type(beta))*cosh(-0.5*beta*x);
        //return exp(-0.5*scalar_type(beta))*cosh(-0.5*beta*x);
    };

    std::vector<mpreal> section_edges;
    for (int s=0; s<basis.ul(0).num_sections()+1; ++s) {
        section_edges.push_back(-basis.ul(0).section_edge(
                basis.ul(0).num_sections()-s
        ));
    }
    for (int s=0; s<basis.ul(0).num_sections()+1; ++s) {
        section_edges.push_back(basis.ul(0).section_edge(s));
    }
    //auto section_edges = irlib::linspace<scalar_type>(-1, 1, 50000);

    std::vector<double> coeff(basis.dim());
    for (int l = 0; l < basis.dim(); ++l) {
      auto f = [&](const accurate_fp_type& x) {return accurate_fp_type(gtau(x) * basis.ulx_mp(l,x));};
      coeff[l] = static_cast<double>(irlib::integrate_gauss_legendre<mpreal,accurate_fp_type>(section_edges, f, 12) * beta / std::sqrt(2.0));
    }

    std::vector<double> y_r(nptr, 0.0);
    for (int l = 0; l < basis.dim(); ++l) {
      for (int i = 0; i < nptr; ++i) {
        y_r[i] += coeff[l] * (std::sqrt(2.0)/beta) * basis.ulx_mp(l,x[i]);
      }
    }

    double max_diff = 0.0;
    for (int i = 0; i < nptr; ++i) {
        //std::cout << x[i] << " " << gtau(x[i]) << " " << y_r[i] << " " << gtau(x[i])-y_r[i] << std::endl;
      max_diff = std::max(
                      std::abs(static_cast<double>(gtau(x[i])-y_r[i])),
                      max_diff);
    }
    ASSERT_NEAR(max_diff, 0.0, tol);

    //to matsubara freq.
    std::vector<long> n_vec;
    for (int n=0; n<1000; ++n) {
        n_vec.push_back(n);
    }
    //some higher frequencies
    n_vec.push_back(1000000);
    n_vec.push_back(100000000);
    int n_iw = n_vec.size();
    MatrixXc Tnl(n_iw, basis.dim());
    auto Tnl_tensor = basis.compute_Tnl(n_vec);
    MatrixXc coeff_vec(basis.dim(),1);
    for (int l = 0; l < basis.dim(); ++l) {
      coeff_vec(l,0) = coeff[l];
    }
    MatrixXc coeff_iw = Eigen::Map<MatrixXc>(&Tnl_tensor(0,0), n_vec.size(), basis.dim()) * coeff_vec;

    const std::complex<double> zi(0.0, 1.0);
    for (int n = 0; n < n_iw; ++n) {
      double wn = (2.*n_vec[n]+1)*M_PI/beta;
      std::complex<double> z = - 0.5/(zi*wn - 1.0) - 0.5/(zi*wn + 1.0);
      ASSERT_NEAR(z.real(), coeff_iw(n).real(), tol);
      ASSERT_NEAR(z.imag(), coeff_iw(n).imag(), tol);
        //std::cout << " n " << n << " " << z.imag() << " " << z.imag() - coeff_iw(n).imag() << std::endl;
    }

  }
}

