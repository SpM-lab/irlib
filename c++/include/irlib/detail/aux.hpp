#pragma once

#include <algorithm>

#include <Eigen/Core>
#include <Eigen/CXX11/Tensor>

#include "../piecewise_polynomial.hpp"
#include "spline.hpp"

namespace irlib {
    template<typename mp_type>
    std::vector<std::pair<mp_type, mp_type> >
    composite_gauss_legendre_nodes(
            const std::vector<mp_type> &section_edges,
            const std::vector<std::pair<mp_type, mp_type> > &nodes
    ) {
        int num_sec = section_edges.size() - 1;
        int num_local_nodes = nodes.size();

        std::vector<std::pair<mp_type, mp_type> > all_nodes(num_sec * num_local_nodes);
        for (int s = 0; s < num_sec; ++s) {
            auto a = section_edges[s];
            auto b = section_edges[s + 1];
            for (int n = 0; n < num_local_nodes; ++n) {
                mp_type x = a + ((b - a) / mp_type(2)) * (nodes[n].first + mp_type(1));
                mp_type w = ((b - a) / mp_type(2)) * nodes[n].second;
                all_nodes[s * num_local_nodes + n] = std::make_pair(x, w);
            }
        }
        return all_nodes;
    };

    template<typename Tx, typename Ty, typename F>
    Ty integrate_gauss_legendre(const std::vector<Tx>& section_edges, const F& f, int num_local_nodes) {
        std::vector<std::pair<mpreal, mpreal >> nodes = detail::gauss_legendre_nodes<mpreal>(num_local_nodes);
        auto nodes_x = composite_gauss_legendre_nodes(section_edges, nodes);
        Ty r = 0;
        for (int n=0; n<nodes_x.size(); ++n) {
            r += f(nodes_x[n].first) * nodes_x[n].second;
        }
        return r;
    };

    template<typename T, typename Tx>
    piecewise_polynomial<T,Tx> construct_piecewise_polynomial_cspline(
            const std::vector<Tx> &x_array, const std::vector<double> &y_array) {
        const int n_points = x_array.size();
        const int n_section = n_points - 1;

        Eigen::MatrixXd coeff(n_section, 4);

        // Cubic spline interpolation
        tk::spline spline;
        std::vector<double> x_array_d(x_array.size());
        for (int i=0; i<x_array.size(); ++i) {
            x_array_d[i] = static_cast<double>(x_array[i]);
        }
        spline.set_points(x_array_d, y_array);

        // Construct piecewise_polynomial
        for (int s = 0; s < n_section; ++s) {
            for (int p = 0; p < 4; ++p) {
                coeff(s, p) = spline.get_coeff(s, p);
            }
        }
        piecewise_polynomial<T,Tx> tmp(n_section, x_array, coeff);
        return piecewise_polynomial<T,Tx>(n_section, x_array, coeff);
    };

    template<typename T>
    inline std::vector<T> linspace(T minval, T maxval, int N) {
        std::vector<T> r(N);
        for (int i = 0; i < N; ++i) {
            r[i] = i * (maxval - minval) / (N - T(1)) + minval;
        }
        return r;
    }

    //Compute nodes (zeros) of Legendre polynomials
    inline std::vector<double> compute_legendre_nodes(int l) {
        double eps = 1e-10;
        if (l > 200) {
            throw std::runtime_error("l > 200 in compute_legendre_nodes");
        }

        std::vector<double> nodes;

        auto leg_diff = [](int l, double x) {
            return l * (x * legendre_p(l, x) - legendre_p(l - 1, x)) / (x * x - 1);
        };

        //i-th zero
        for (int i = 0; i < l / 2; i++) {
            //initial guess
            double x = std::cos(M_PI * (i + 1 - 0.25) / (l + 0.5));

            //Newton-Raphson iteration
            while (true) {
                double leg = legendre_p(l, x);
                double x_new = x - 0.1 * leg / leg_diff(l, x);
                if (std::abs(x_new - x) < eps && std::abs(leg) < eps) {
                    break;
                }
                x = x_new;
            }

            nodes.push_back(x);
            nodes.push_back(-x);
        }

        if (l % 2 == 1) {
            nodes.push_back(0.0);
        }

        std::sort(nodes.begin(), nodes.end());

        return nodes;
    }

    //AVOID USING BOOST_TYPEOF
    template<class T1, class T2>
    struct result_of_overlap {
        typedef std::complex<double> value;
    };

    template<>
    struct result_of_overlap<double, double> {
        typedef double value;
    };

    /*
    template<typename T, typename Tx>
    void compute_integral_with_exp(
            const std::vector<double> &w,
            const std::vector<piecewise_polynomial<T,Tx> > &pp_func,
            Eigen::Tensor<std::complex<double>, 2> &Tnl
    );
    */

    /// Construct piecewise polynomials representing exponential functions: exp(i w_i x)
    template<class T, typename Tx>
    void construct_exp_functions_coeff(
            const std::vector<double> &w,
            const std::vector<Tx> &section_edges,
            int k,
            Eigen::Tensor<std::complex<T>, 3> &coeffs) {
        const int N = section_edges.size() - 1;

        std::complex<double> z;
        coeffs = Eigen::Tensor<std::complex<T>,3>(w.size(), N, k + 1);

        std::vector<double> pre_factor(k + 1);
        pre_factor[0] = 1.0;
        for (int j = 1; j < k + 1; ++j) {
            pre_factor[j] = pre_factor[j - 1] / j;
        }

        for (int n = 0; n < w.size(); ++n) {
            auto z = std::complex<double>(0.0, w[n]);
            for (int section = 0; section < N; ++section) {
                const double x = static_cast<double>(section_edges[section]);
                std::complex<T> exp0 = std::exp(z * (x + 1));
                std::complex<T> z_power = 1.0;
                for (int j = 0; j < k + 1; ++j) {
                    coeffs(n, section, j) = exp0 * z_power * pre_factor[j];
                    z_power *= z;
                }
            }
        }
    }

/**
 *  Compute \int_x0^x1 dx exp(i w (x+1)) (x-x0)^k
 *      for k=0, 1, ..., K+1. x1 = x0+dx.
**/
    inline void compute_Ik(double x0, double dx, double w, int K, std::vector<std::complex<double> >& Ik) {
        auto x1 = x0 + dx;
        auto iw = std::complex<double>(0.0, w);
        auto exp0 = std::exp(iw*(x0+1));
        auto exp1 = std::exp(iw*(x1+1));
        Ik[0] = (exp1 - exp0)/iw;

        auto dx_k = dx;
        for (int k=1; k<K+1; ++k) {
            Ik[k] = (dx_k * exp1 - (k * 1.0) * Ik[k-1])/iw;
            dx_k *= dx;
        }
    }

/**
 * Compute integral of exponential functions and given piecewise polynomials
 *           \int dx exp(i w_i (x+1)) p_j(x),
 *           where w_i are given real double objects and p_j are piecewise polynomials.
 * @tparam T  scalar type of piecewise polynomials
 * @param w vector of w_i in ascending order
 * @param statis Statistics (fermion or boson)
 * @param p vector of piecewise polynomials.
 * @param results  computed results
 */
    template<typename T, typename Tx>
    void compute_integral_with_exp(
            const std::vector<double> &w,
            const std::vector<piecewise_polynomial<T,Tx> > &pp_func,
            Eigen::Tensor<std::complex<double>, 2> &Tnl
    ) {
        typedef std::complex<double> dcomplex;
        typedef piecewise_polynomial<std::complex<double>,Tx> pp_type;
        typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
        typedef Eigen::Tensor<std::complex<double>, 2> tensor_t;

        //order of polynomials used for representing exponential functions internally.
        const int k_iw = 16;//for debug
        const int k = pp_func[0].order();
        const int n_section = pp_func[0].num_sections();

        std::vector<std::complex<double> > Ik(k+1);

        for (int l = 0; l < pp_func.size(); ++l) {
            if (k != pp_func[l].order()) {
                throw std::runtime_error(
                        "Error in compute_transformation_matrix_to_matsubara: basis functions must be pieacewise polynomials of the same order");
            }
            if (pp_func[l].section_edge(0) != 0 || pp_func[l].section_edge(n_section) != 1) {
                throw std::runtime_error("Piecewise polynomials must be defined on [0,1]");
            }
        }

        const int n_iw = w.size();
        const int n_max = n_iw - 1;

        for (int i = 0; i < w.size() - 1; ++i) {
            if (w[i] > w[i + 1]) {
                throw std::runtime_error("w must be give in ascending order.");
            }
        }

        //Use Taylor expansion for exp(i w_n tau) for w_n*dx < cutoff*M_PI
        const double cutoff = 0.1;

        Eigen::Tensor<std::complex<double>,3> exp_coeffs(w.size(), n_section, k_iw + 1);
        construct_exp_functions_coeff(w, pp_func[0].section_edges(), k_iw, exp_coeffs);

        matrix_t left_mid_matrix(n_iw, k + 1);
        matrix_t left_matrix(n_iw, k_iw + 1);
        matrix_t mid_matrix(k_iw + 1, k + 1);
        matrix_t right_matrix(k + 1, pp_func.size());
        matrix_t r(n_iw, pp_func.size());
        r.setZero();

        std::vector<double> dx_power(k + k_iw + 2);

        for (int s = 0; s < n_section; ++s) {
            auto x0 = pp_func[0].section_edge(s);
            double dx = static_cast<double>(pp_func[0].section_edge(s + 1) - pp_func[0].section_edge(s));
            left_mid_matrix.setZero();

            dx_power[0] = 1.0;
            for (int p = 1; p < dx_power.size(); ++p) {
                dx_power[p] = dx * dx_power[p - 1];
            }

            //Use Taylor expansion for exp(i w_n tau) for w_n*dx < cutoff*M_PI
            const double w_max_cs = cutoff * M_PI / dx;
            int n_max_cs = -1;
            for (int i = 0; i < w.size(); ++i) {
                if (w[i] <= w_max_cs) {
                    n_max_cs = i;
                }
            }

            //Use Taylor expansion
            if (n_max_cs >= 0) {
                for (int p = 0; p < k_iw + 1; ++p) {
                    for (int p2 = 0; p2 < k + 1; ++p2) {
                        mid_matrix(p, p2) = dx_power[p + p2 + 1] / (p + p2 + 1.0);
                    }
                }

                for (int n = 0; n < n_max_cs + 1; ++n) {
                    for (int p = 0; p < k_iw + 1; ++p) {
                        left_matrix(n, p) = exp_coeffs(n,s,p);
                    }
                }

                left_mid_matrix.block(0, 0, n_max_cs + 1, k + 1) =
                        left_matrix.block(0, 0, n_max_cs + 1, k_iw + 1) * mid_matrix;
            }

            //Otherwise, compute the overlap exactly
            for (int n = std::max(n_max_cs + 1, 0); n <= n_max; ++n) {
                compute_Ik(static_cast<double>(x0), dx, w[n], k, Ik);
                for (int i=0; i<k+1; ++i) {
                    left_mid_matrix(n, i) = Ik[i];
                }
            }

            for (int l = 0; l < pp_func.size(); ++l) {
                for (int p2 = 0; p2 < k + 1; ++p2) {
                    right_matrix(p2, l) = pp_func[l].coefficient(s, p2);
                }
            }

            r += left_mid_matrix * right_matrix;
        }

        Tnl = tensor_t(n_iw, pp_func.size());
        for (int n = 0; n < n_iw; ++n) {
            for (int l = 0; l < pp_func.size(); ++l) {
                Tnl(n, l) = r(n, l);
            }
        }
    }


    /**
    * Compute a transformation matrix from a give orthogonal basis set to Matsubara freq.
    * @tparam T  scalar type
    * @param n_vec indices of Matsubara frequqneices for which matrix elements will be computed (in strictly ascending order).
    *          The Matsubara basis functions look like exp(i PI * (n[i]+1/2)) for fermions, exp(i PI * n[i]) for bosons.
    * @param bf_src orthogonal basis functions on [-1,1]. They must be piecewise polynomials of the same order. Piecewise polynomial representations on [0,1] must be provided.
    *               Basis functions u_l(x) are assumed to be even or odd for even l and odd l, respectively.
    * @param Tnl  computed transformation matrix
    */
    template<typename T, typename Tx>
    void compute_transformation_matrix_to_matsubara(
            const std::vector<long> &n_vec,
            irlib::statistics::statistics_type statis,
            const std::vector<piecewise_polynomial<T,Tx>> &bf_src,
            Eigen::Tensor<std::complex<double>, 2> &Tnl
    ) {
        typedef std::complex<double> dcomplex;
        typedef piecewise_polynomial<std::complex<double>,Tx> pp_type;
        typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
        typedef Eigen::Tensor<std::complex<double>, 2> tensor_t;

        long n_limit = 500;

        if (n_vec.size() == 0) {
            return;
        }

        for (int i = 0; i < n_vec.size() - 1; ++i) {
            if (n_vec[i] > n_vec[i + 1]) {
                throw std::runtime_error("n_vec must be in strictly ascending order!");
            }
        }

        if (n_vec[0] < 0) {
            throw std::runtime_error("n_vec cannot be negative!");
        }

        std::vector<long> ovec;
        long offset = (statis == statistics::FERMIONIC ? 1 : 0);
        for (auto n : n_vec) {
            if (n < n_limit) {
                ovec.push_back(2*n+offset);
            }
        }

        Eigen::Tensor<std::complex<double>, 2> Tnl_low_freq;
        compute_Tbar_ol(ovec, bf_src, Tnl_low_freq);

        Tnl = Eigen::Tensor<std::complex<double>,2>(n_vec.size(), bf_src.size());
        Tnl.setZero();
        for(int j=0; j<Tnl_low_freq.dimension(1); ++j) {
            for(int i=0; i<Tnl_low_freq.dimension(0); ++i) {
                Tnl(i,j) = Tnl_low_freq(i,j);
            }
        }


        int sign_s = (statis == statistics::FERMIONIC ? 1 : -1);
        for (int l=0; l<bf_src.size(); ++l) {
            for (int m=0; m<bf_src[l].order(); ++m) {
                int sign_m1 = (l+m)%2==0 ? 1 : -1;
                if (sign_s == sign_m1) {
                    double am = std::sqrt(2.0) * std::pow(2, m) * (sign_s + sign_m1) * bf_src[l].derivative(1, m);
                    for (int i=ovec.size(); i<n_vec.size(); ++i) {
                        double wn = (2*n_vec[i]+offset) * M_PI;
                        Tnl(i, l) += am/std::pow(std::complex<double>(0.0, -wn), m+1);
                    }
                }
            }
        }
    }

    /**
    * Compute a transformation matrix (\bar{T}_{nl}) from a give orthogonal basis set to Matsubara freq.
    * @tparam T  scalar type
    * @param n indices of Matsubara frequqneices for which matrix elements will be computed (in strictly ascending order).
    *          The Matsubara basis functions look like exp(i PI * (n[i]/2) * (x+1)).
    * @param bf_src orthogonal basis functions on [-1,1]. They must be piecewise polynomials of the same order. Piecewise polynomial representations on [0,1] must be provided.
    *               Basis functions u_l(x) are assumed to be even or odd for even l and odd l, respectively.
    * @param Tnl  computed transformation matrix
    */
    template<typename T, typename Tx>
    void compute_Tbar_ol(
            const std::vector<long> &o_vec,
            const std::vector<irlib::piecewise_polynomial<T,Tx>> &bf_src,
            Eigen::Tensor<std::complex<double>, 2> &Tbar_ol
    ) {
        typedef std::complex<double> dcomplex;
        typedef piecewise_polynomial<std::complex<double>,Tx> pp_type;
        typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
        typedef Eigen::Tensor<std::complex<double>, 2> tensor_t;

        if (o_vec.size() == 0) {
            return;
        }

        for (int i = 0; i < o_vec.size() - 1; ++i) {
            if (o_vec[i] > o_vec[i + 1]) {
                throw std::runtime_error("o must be given in strictly ascending order!");
            }
        }

        std::vector<double> w;
        std::transform(o_vec.begin(), o_vec.end(), std::back_inserter(w), [](long o) { return 0.5 * M_PI * o; });

        compute_integral_with_exp(w, bf_src, Tbar_ol);

        for (int l=0; l<bf_src.size(); ++l) {
            for (int i=0; i<o_vec.size(); ++i) {
                if ( (l+o_vec[i])%2 == 0) {
                    Tbar_ol(i,l) = 2 * Tbar_ol(i,l).real();
                } else {
                    Tbar_ol(i,l) = std::complex<double>(0.0, 2 * Tbar_ol(i,l).imag());
                }
            }
        }

        std::vector<double> inv_norm(bf_src.size());
        for (int l = 0; l < bf_src.size(); ++l) {
            inv_norm[l] = 1. / std::sqrt(2*static_cast<double>(bf_src[l].overlap(bf_src[l])));
        }
        for (int n = 0; n < w.size(); ++n) {
            for (int l = 0; l < bf_src.size(); ++l) {
                Tbar_ol(n, l) *= inv_norm[l] * std::sqrt(0.5);
            }
        }
    }


    /// Compute overlap <left | right> with complex conjugate
    template<typename T1, typename T2, typename Tx>
    void compute_overlap(
            const std::vector<piecewise_polynomial<T1,Tx> > &left_vectors,
            const std::vector<piecewise_polynomial<T2,Tx> > &right_vectors,
            Eigen::Matrix<typename result_of_overlap<T1, T2>::value, Eigen::Dynamic, Eigen::Dynamic> &results) {
        typedef typename result_of_overlap<T1, T2>::value Tr;

        const int NL = left_vectors.size();
        const int NR = right_vectors.size();
        const int n_sections = left_vectors[0].num_sections();

        const int k1 = left_vectors[0].order();
        const int k2 = right_vectors[0].order();

        if (left_vectors[0].section_edges() != right_vectors[0].section_edges()) {
            throw std::runtime_error("Not supported");
        }

        for (int n = 0; n < NL - 1; ++n) {
            if (left_vectors[n].section_edges() != left_vectors[n + 1].section_edges()) {
                throw std::runtime_error("Not supported");
            }
        }

        for (int n = 0; n < NL; ++n) {
            if (k1 != left_vectors[n].order()) {
                throw std::runtime_error("Left vectors must be piecewise polynomials of the same order.");
            }
        }

        for (int n = 0; n < NR; ++n) {
            if (k2 != right_vectors[n].order()) {
                throw std::runtime_error("Right vectors must be piecewise polynomials of the same order.");
            }
        }

        for (int l = 0; l < NR - 1; ++l) {
            if (right_vectors[l].section_edges() != right_vectors[l + 1].section_edges()) {
                throw std::runtime_error("Not supported");
            }
        }

        std::vector<double> x_min_power(k1 + k2 + 2), dx_power(k1 + k2 + 2);

        Eigen::Matrix<Tr, Eigen::Dynamic, Eigen::Dynamic> mid_matrix(k1 + 1, k2 + 1);
        Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> left_matrix(NL, k1 + 1);
        Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic> right_matrix(k2 + 1, NR);
        results.resize(NL, NR);

        results.setZero();
        for (int s = 0; s < n_sections; ++s) {
            dx_power[0] = 1.0;
            const double dx = left_vectors[0].section_edge(s + 1) - left_vectors[0].section_edge(s);
            for (int p = 1; p < dx_power.size(); ++p) {
                dx_power[p] = dx * dx_power[p - 1];
            }

            for (int p = 0; p < k1 + 1; ++p) {
                for (int p2 = 0; p2 < k2 + 1; ++p2) {
                    mid_matrix(p, p2) = dx_power[p + p2 + 1] / (p + p2 + 1.0);
                }
            }

            for (int n = 0; n < NL; ++n) {
                for (int p = 0; p < k1 + 1; ++p) {
                    left_matrix(n, p) = std::conj(left_vectors[n].coefficient(s, p));
                }
            }

            for (int l = 0; l < NR; ++l) {
                for (int p2 = 0; p2 < k2 + 1; ++p2) {
                    right_matrix(p2, l) = right_vectors[l].coefficient(s, p2);
                }
            }

            results += left_matrix * (mid_matrix * right_matrix);
        }

    }


    /**
     * Find approximate positions of nodes for the even singular vectors with the lowest singular value larger than a cutoff
     * @tparam Kernel kernel type
     * @param knl Kernel object
     * @param N Number of points for discretizing the kernel
     * @param cutoff_singular_values smallest relative singular value
     * @return positions of nodes
     */
    template<typename Kernel>
    std::pair<std::vector<double>,std::vector<double>>
    compute_approximate_nodes_even_sector(const Kernel &knl, int N, double cutoff_singular_values) {
        typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

        double de_cutoff = 2.5;

        //DE mesh for x
        std::vector<double> tx_vec = linspace<double>(0.0, de_cutoff, N);
        std::vector<double> weight_x(N), x_vec(N);
        for (int i = 0; i < N; ++i) {
            x_vec[i] = std::tanh(0.5 * M_PI * std::sinh(tx_vec[i]));
            //sqrt of the weight of DE formula
            weight_x[i] =
                    std::sqrt(0.5 * M_PI * std::cosh(tx_vec[i])) / std::cosh(0.5 * M_PI * std::sinh(tx_vec[i]));
        }

        //DE mesh for y
        std::vector<double> ty_vec = linspace<double>(-de_cutoff, 0.0, N);
        std::vector<double> y_vec(N), weight_y(N);
        for (int i = 0; i < N; ++i) {
            y_vec[i] = std::tanh(0.5 * M_PI * std::sinh(ty_vec[i])) + 1.0;
            //sqrt of the weight of DE formula
            weight_y[i] =
                    std::sqrt(0.5 * M_PI * std::cosh(ty_vec[i])) / std::cosh(0.5 * M_PI * std::sinh(ty_vec[i]));
        }

        matrix_t K(N, N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                K(i, j) = weight_x[i] * static_cast<double>(knl(x_vec[i], y_vec[j]) + knl(x_vec[i], -y_vec[j])) *
                          weight_y[j];
            }
        }

        //Perform SVD
        //Eigen::VectorXd svalues(N);
        //matrix_t U(N, N), Vt(N, N);
        //svd_square_matrix(K, N, svalues, Vt, U);
        Eigen::BDCSVD<matrix_t> svd(K, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::VectorXd& svalues = svd.singularValues();
        const matrix_t& U = svd.matrixU();
        const matrix_t& V = svd.matrixV();

        //Count non-zero SV
        int dim = N;
        for (int i = 1; i < N; ++i) {
            if (std::abs(svalues(i) / svalues(0)) < cutoff_singular_values) {
                dim = i;
                break;
            }
        }

        //find nodes
        std::vector<double> nodes_x, nodes_y;
        for (int i = 0; i < N - 1; ++i) {
            if (U(i, dim - 1) * U(i + 1, dim - 1) < 0.0) {
                nodes_x.push_back(0.5 * (x_vec[i] + x_vec[i + 1]));
            }
            if (V(i, dim - 1) * V(i+1, dim - 1) < 0.0) {
                nodes_y.push_back(0.5 * (y_vec[i] + y_vec[i + 1]));
            }
        }

        if (nodes_x.size() != dim - 1 || nodes_y.size() != dim - 1) {
            throw std::runtime_error("The number of nodes is wrong.");
        }

        return std::make_pair(nodes_x, nodes_y);
    }
}
