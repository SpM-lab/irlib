#include "common.hpp"

#include <fstream>

using namespace irlib;


//inline rerror(const std::complex<mpreal>& z1, const std::complex<mpreal>& z2) {
    //auto diff = z1-z2;
//}
inline std::complex<double>
to_dcomplex(std::complex<mpreal>& z) {
    return std::complex<double>(static_cast<double>(z.real()), static_cast<double>(z.imag()));
}

TEST(precomputed_basis, double_precision) {
    auto b = loadtxt("./samples/np10/basis_f-mp-Lambda10000.0.txt");
    ir_set_default_prec<mpreal>(b.get_prec());
    auto dim = b.dim();

    std::vector<piecewise_polynomial<mpreal,mpreal> > basis_vectors_org;
    std::vector<double> sl;
    for (int l=0; l<dim; ++l) {
        basis_vectors_org.push_back(b.ul(l));
        sl.push_back(b.sl(l)/b.sl(0));
    }

    auto xvec = linspace<double>(0.99, 1, 1000);
    for (auto x : xvec) {
        int l = dim -1;
        auto diff = basis_vectors_org[l].template compute_value<mpreal>(x) - basis_vectors_org[l].template compute_value<double>(x);
        ASSERT_TRUE(std::abs(static_cast<double>(diff)) < 1e-10);
    }
}


TEST(precomputed_basis, Tnl) {
    int num_local_nodes = 4*48;

    auto b = loadtxt("./samples/np10/basis_f-mp-Lambda10000.0.txt");
    ir_set_default_prec<mpreal>(b.get_prec());
    auto dim = b.dim();
    std::vector<mpreal> section_edges = b.ul(dim-1).section_edges();
    auto local_nodes = detail::gauss_legendre_nodes<mpreal>(num_local_nodes);
    auto global_nodes = composite_gauss_legendre_nodes(section_edges, local_nodes);

    //std::vector<long> n_vec {0, 10, 100, 1000};
    std::vector<long> n_vec {0, 1000};

    auto Tnl = b.compute_Tnl(n_vec);
    auto Tnl_ref = MatrixXc(n_vec.size(), b.dim());

    int index_n = 0;
    for (auto n : n_vec) {
        int o = 2*n+1;
        std::complex<mpreal> z(0.0, mpfr::const_pi() * 0.5 * o);

        using MatrixXcmp = Eigen::Matrix<std::complex<mpreal>,Eigen::Dynamic,Eigen::Dynamic>;

        MatrixXcmp left_mat(1, global_nodes.size());
        for (int p=0; p<global_nodes.size(); ++p) {
            left_mat(0,p) = global_nodes[p].second * my_exp(z.imag()*global_nodes[p].first);
        }
        MatrixXcmp right_mat(global_nodes.size(), b.dim());
        for (int l=0; l<b.dim(); ++l) {
            for (int p=0; p<global_nodes.size(); ++p) {
                right_mat(p,l) = b.ulx_mp(l, global_nodes[p].first);
            }
        }
        MatrixXcmp  Aol = left_mat * right_mat;

        auto exp_z = my_exp(z.imag());
        for (int l=0; l<b.dim(); ++l) {
            std::complex<mpreal> r;
            if (l%2 == 0) {
                r = exp_z*mpfr::sqrt(2) * Aol(0,l).real();
            } else {
                r = exp_z*mpfr::sqrt(2) * std::complex<mpreal>(0, Aol(0,l).imag());
            }
            Tnl_ref(index_n,l) = to_dcomplex(r);

            std::complex<mpreal> Tnl_new = compute_Tnl_impl(b.ul(l), l%2==0, irlib::statistics::FERMIONIC, mpfr::const_pi() * 0.5 * o);
            std::complex<double> Tnl_safe = b.compute_Tnl_safe(n, l);

            ASSERT_TRUE(std::abs(to_dcomplex(r)-Tnl(index_n,l))/std::abs(Tnl(index_n,l)) < 1e-5);
            ASSERT_TRUE(std::abs(to_dcomplex(Tnl_new)-Tnl(index_n,l))/std::abs(Tnl(index_n,l)) < 1e-5);
            ASSERT_TRUE(std::abs(to_dcomplex(Tnl_new)-Tnl_safe)/std::abs(Tnl_safe) < 1e-10);
        }
        ++ index_n;
    }

    // check tail
    /*
    for (auto n : std::vector<long>{1000000000}) {
        auto o = 2*n+1;
        auto w = mpfr::const_pi() * 0.5 * o;
        auto l = b.dim()-1;
        auto Tnl3 = b.compute_Tnl_safe(n, l);
    }
    */

}

TEST(precomputed_basis, derivatives) {
    using namespace irlib;

    for (auto s : std::vector<statistics::statistics_type>{statistics::FERMIONIC, statistics::BOSONIC}) {
        std::string str_s = s == statistics::FERMIONIC ? "f" : "b";
        auto b_np8 = loadtxt("./samples/np8/basis_"+str_s+"-mp-Lambda10000.0.txt");
        auto b_np10 = loadtxt("./samples/np10/basis_"+str_s+"-mp-Lambda10000.0.txt");
        auto dim = b_np8.dim();

        for (int order = 0; order <= 2; ++order) {
            auto rat = b_np8.ulx_derivative(dim-1, 1.0, order)/b_np10.ulx_derivative(dim-1, 1.0, order);
            //std::cout << str_s << order << " " << rat-1 << std::endl;
            ASSERT_NEAR(rat, 1.0, 1e-8);
        }
    }
}

TEST(precomputed_basis, Tnl_high_freq_limit) {
    using namespace irlib;

    for (auto s : std::vector<statistics::statistics_type>{statistics::FERMIONIC, statistics::BOSONIC}) {
        std::string str_s = (s == statistics::FERMIONIC ? "f" : "b");
        auto b = loadtxt("./samples/np8/basis_"+str_s+"-mp-Lambda10000.0.txt");
        auto dim = b.dim();

        long n = 1E+10;
        double shift = (s == statistics::FERMIONIC ? 0.5 : 0.0);

        int sign_s = (s == statistics::FERMIONIC ? -1 : 1);

        for (int l = 0; l < dim; ++l) {
            std::complex<double> Tnl_safe = b.compute_Tnl_safe(n, l);
            std::complex<double> Tnl_ref = 0.0;
            std::complex<double> fact = static_cast<double>(sign_s)/std::complex<double>(0.0, M_PI * (n + shift));

            std::complex<double> ztmp = fact;
            for (int m=0; m < 2; ++m) {
                int sign_lm = ((l+m)%2 == 0 ? 1 : -1);
                Tnl_ref += ztmp * static_cast<double>(1 - sign_s * sign_lm) * b.ulx_derivative(l, 1.0, m);
                ztmp *= fact;
            }
            Tnl_ref /= std::sqrt(2.0);

            //std::cout << " " << l << " " << Tnl_safe << " " << Tnl_ref << " " << std::abs(Tnl_safe/Tnl_ref) << std::endl;
            ASSERT_NEAR(std::abs(Tnl_safe/Tnl_ref), 1.0, 1e-8);
        }

    }
}
