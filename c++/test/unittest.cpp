#include "unittest.hpp"

#include <fstream>

using namespace irlib;

TEST(PiecewisePolynomial, Orthogonalization) {
    typedef double Scalar;
    const int n_section = 10, k = 8, n_basis = 3;
    typedef irlib::piecewise_polynomial<Scalar,mpreal> pp_type;

    std::vector<mpreal> section_edges(n_section+1);
    boost::multi_array<Scalar,3> coeff(boost::extents[n_basis][n_section][k+1]);

    for (int s = 0; s < n_section + 1; ++s) {
        section_edges[s] = s*2.0/n_section - 1.0;
    }
    section_edges[0] = -1.0;
    section_edges[n_section] = 1.0;

    std::vector<pp_type> nfunctions;

    // x^0, x^1, x^2, ...
    for (int n = 0; n < n_basis; ++ n) {
        boost::multi_array<Scalar,2> coeff(boost::extents[n_section][k+1]);
        std::fill(coeff.origin(), coeff.origin()+coeff.num_elements(), 0.0);

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
                coeff[s][l] = static_cast<double>(rtmp * pow(section_edges[s], n-l));
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

    //check transformation matrix to Matsubara frequencies
    if (basis.get_statistics() == irlib::statistics::FERMIONIC) {

      const int N_iw = 3;
      //time_t t1, t2;

      boost::multi_array<std::complex<double>,2> Tnl_legendre(boost::extents[N_iw][3]);

      compute_Tnl_legendre(N_iw, 3, Tnl_legendre);

      //Fast version
      std::vector<long> n_vec;
      for (int n=0; n<N_iw; ++n) {
          n_vec.push_back(n);
      }
      auto Tnl_ir = basis.compute_Tnl(n_vec);
      for (int n = 0; n < N_iw; n++) {
        for (int l = 0; l < 3; ++l) {
          ASSERT_NEAR(std::abs(Tnl_ir(n,l) / (Tnl_legendre[n][l]) - 1.0), 0.0, 1e-5);
        }
      }
    }
  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

TEST(IrBasis, FermionInsulatingGtau) {
  try {
    const double Lambda = 300.0, beta = 100.0;
    const int max_dim = 100;
    irlib::basis_f basis(Lambda, max_dim, 1e-14);
    ASSERT_TRUE(basis.dim()>0);

    typedef irlib::piecewise_polynomial<double,mpreal> pp_type;

      std::cout << "dim " << basis.dim() << std::endl;
      std::cout << "section " << basis.ul(0).num_sections() << std::endl;

    const int nptr = basis.ul(0).num_sections() + 1;
    std::vector<mpreal> x(nptr);
    for (int i = 0; i < nptr; ++i) {
      x[i] = basis.ul(0).section_edge(i);
    }

    auto gtau = [&](const mpreal& x) {return std::exp(-0.5*beta)*cosh(-0.5*beta*x);};
    auto section_edges = irlib::linspace<mpfr::mpreal>(-1, 1, 500);

    std::vector<double> coeff(basis.dim());
    for (int l = 0; l < basis.dim(); ++l) {
      auto f = [&](const mpreal& x) {return mpreal(gtau(x) * basis.ulx(l,x));};
      coeff[l] = static_cast<double>(irlib::integrate_gauss_legendre<mpreal,mpreal>(section_edges, f, 12) * beta / std::sqrt(2.0));
    }

    std::vector<double> y_r(nptr, 0.0);
    for (int l = 0; l < basis.dim(); ++l) {
      for (int i = 0; i < nptr; ++i) {
        y_r[i] += coeff[l] * (std::sqrt(2.0)/beta) * basis.ulx(l,x[i]);
      }
    }

    double max_diff = 0.0;
    for (int i = 0; i < nptr; ++i) {
      max_diff = static_cast<double>(max(abs(gtau(x[i])-y_r[i]), max_diff));
      ASSERT_TRUE(abs(gtau(x[i])-y_r[i]) < 1e-8);
    }

    //to matsubara freq.
    const int n_iw = 1000;
    boost::multi_array<std::complex<double>,2> Tnl(boost::extents[n_iw][basis.dim()]);
    std::vector<long> n_vec;
    for (int n=0; n<n_iw; ++n) {
        n_vec.push_back(n);
    }
    auto Tnl_tensor = basis.compute_Tnl(n_vec);
    MatrixXc coeff_vec(basis.dim(),1);
    for (int l = 0; l < basis.dim(); ++l) {
      coeff_vec(l,0) = coeff[l];
    }
    MatrixXc coeff_iw = Eigen::Map<MatrixXc>(&Tnl_tensor(0,0), n_vec.size(), basis.dim()) * coeff_vec;

    const std::complex<double> zi(0.0, 1.0);
    for (int n = 0; n < n_iw; ++n) {
      double wn = (2.*n+1)*M_PI/beta;
      std::complex<double> z = - 0.5/(zi*wn - 1.0) - 0.5/(zi*wn + 1.0);
      ASSERT_NEAR(std::abs(z-coeff_iw(n)), 0.0, 1e-9);
    }


  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

