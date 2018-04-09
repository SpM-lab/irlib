#include "common.hpp"

#include <fstream>

using namespace irlib;

inline std::complex<mpreal> my_exp(const mpreal& z_img) {
    return std::complex<mpreal>(
            mpfr::cos(z_img), mpfr::sin(z_img)
    );
}

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

    /*
    auto basis_vectors_cs =  cspline_approximation(basis_vectors_org, 1e-8);

    for (int l=0; l<dim; ++l) {
        for (auto x : xvec) {
            auto diff = basis_vectors_org[l].compute_value(x) -  basis_vectors_cs[l].compute_value(x);
            std::cout << " diff " << l << " " << x << " " << diff << std::endl;
            ASSERT_TRUE(std::abs(static_cast<double>(diff)) < 1e-5);
        }
    }
    */
}

/**
 *  Compute \int_{-1}^1 dx exp(i w x) p(x)
**/
inline std::complex<mpfr::mpreal> compute_Tnl_impl(const piecewise_polynomial<mpreal,mpreal>& p, bool even, mpreal w,
                                     int digits_A = 30, int digits_B = 30) {
    int num_local_nodes = 24;
    auto local_nodes = detail::gauss_legendre_nodes<mpreal>(num_local_nodes);
    std::vector<mpreal> section_edges = p.section_edges();
    auto global_nodes = composite_gauss_legendre_nodes(section_edges, local_nodes);
    auto n_local_nodes = local_nodes.size();

    std::complex<mpreal> result(0);
    for (int s = 0; s < p.num_sections(); ++s) {
        mpreal x0 = p.section_edge(s);
        mpreal x1 = p.section_edge(s+1);

        if (w * (x1-x0) < 0.1 * const_pi<mpreal>()){
            mpfr::mpreal::set_default_prec(mpfr::digits2bits(digits_A));
            std::complex<mpreal> tmp(0);
            for (int n = 0; n < n_local_nodes; ++n) {
                auto x_smpl = global_nodes[s*n_local_nodes + n].first;
                tmp += p.compute_value(x_smpl) * my_exp(w*x_smpl) * global_nodes[s*n_local_nodes + n].second;
            }
            result += tmp;
        } else {
            mpfr::mpreal::set_default_prec(mpfr::digits2bits(digits_B));
            std::complex<mpreal> Jk(0, 0);
            std::complex<mpreal> iw(0, w);
            std::complex<mpreal> exp0 = my_exp(w*x0);
            std::complex<mpreal> exp_tmp = my_exp(w*(x1-x0));

            //p contains x^0, x^1, xj^2, ..., x^K (K = p.order())
            for (int k=p.order(); k >= 0; --k) {
                mpreal f0 = p.derivative(x0, k, s);
                mpreal f1 = p.derivative(x1, k, s);
                Jk = ((exp_tmp * f1 - f0) * exp0 - Jk)/iw;
            }
            result += Jk;
        }
    }

    if (even) {
        return mpfr::sqrt(2) * result.real() * my_exp(w);
    } else {
        return mpfr::sqrt(2) * std::complex<mpreal>(0, result.imag()) * my_exp(w);
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

            std::complex<mpreal> Tnl_new = compute_Tnl_impl(b.ul(l), l%2==0, mpfr::const_pi() * 0.5 * o);

            /*
            std::cout << " n " << n << " l " << l << " " << std::abs(to_dcomplex(r)-Tnl(index_n,l))/std::abs(Tnl(index_n,l))
                      << "      " << r.real() << " " << r.imag() << "    " << Tnl(index_n,l).real() << " " << Tnl(index_n,l).imag()
                      << "      " << Tnl_new.real() << " " << Tnl_new.imag()
                      << std::endl;
                      */
            ASSERT_TRUE(std::abs(to_dcomplex(r)-Tnl(index_n,l))/std::abs(Tnl(index_n,l)) < 1e-5);
            ASSERT_TRUE(std::abs(to_dcomplex(Tnl_new)-Tnl(index_n,l))/std::abs(Tnl(index_n,l)) < 1e-5);
        }
        ++ index_n;
    }

for (auto n : std::vector<long>{0,1,10,100,1000,10000,100000, 1000000, 10000000}) {
    /*
    for (int n = 0; n < 10000; ++n) {*/
        auto o = 2*n+1;
        auto l = b.dim()-1;
        auto Tnl = compute_Tnl_impl(b.ul(l), l%2==0, mpfr::const_pi() * 0.5 * o, 15, 15);
        auto Tnl3 = compute_Tnl_impl(b.ul(l), l%2==0, mpfr::const_pi() * 0.5 * o, 20, 20);
        std::cout << "debug n " << n << " " << Tnl.imag() << " " << (Tnl.imag() - Tnl3.imag())/Tnl.imag() << std::endl;
    }

}

