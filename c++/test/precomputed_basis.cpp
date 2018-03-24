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

TEST(precomputed_basis, cubic_spline) {
    auto b = loadtxt("basis_f-mp-Lambda10000.0.txt");
    auto dim = b.dim();

    std::vector<piecewise_polynomial<mpreal,mpreal> > basis_vectors_org;
    for (int l=0; l<dim; ++l) {
        basis_vectors_org.push_back(b.ulx(l));
    }

    auto basis_vectors_cs =  cspline_approximation(basis_vectors, 5);
}

TEST(precomputed_basis, Tnl) {
    int num_local_nodes = 4*48;

    ir_set_default_prec<mpreal>(2*167);

    auto b = loadtxt("basis_f-mp-Lambda10000.0.txt");
    auto dim = b.dim();
    std::vector<mpreal> section_edges = b.ul(dim-1).section_edges();
    auto local_nodes = detail::gauss_legendre_nodes<mpreal>(num_local_nodes);
    auto global_nodes = composite_gauss_legendre_nodes(section_edges, local_nodes);

    std::vector<long> n_vec {0, 10, 100, 1000};

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
            std::cout << " n " << n << " l " << l << " " << std::abs(to_dcomplex(r)-Tnl(index_n,l))/std::abs(Tnl(index_n,l)) << " " << Tnl(index_n,l) << std::endl;
            ASSERT_TRUE(std::abs(to_dcomplex(r)-Tnl(index_n,l))/std::abs(Tnl(index_n,l)) < 1e-5);
        }
        ++ index_n;
    }

}

