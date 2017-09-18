#pragma once

#include <algorithm>

#include <boost/math/special_functions/legendre.hpp>
#include <Eigen/Core>
#include <Eigen/CXX11/Tensor>

#include "../piecewise_polynomial.hpp"

extern "C" void dgesvd_(const char *jobu, const char *jobvt,
                        const int *m, const int *n, double *a, const int *lda,
                        double *s, double *u, const int *ldu,
                        double *vt, const int *ldvt,
                        double *work, const int *lwork, int *info);

extern "C" void dgesdd_(const char *jobz,
                        const int *m, const int *n, double *a, const int *lda,
                        double *s, double *u, const int *ldu,
                        double *vt, const int *ldvt,
                        double *work, const int *lwork, const int *iwork, int *info);

namespace irlib {
    template<typename T>
    irlib::piecewise_polynomial<T> construct_piecewise_polynomial_cspline(
            const std::vector<double> &x_array, const std::vector<double> &y_array);

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
            return l * (x * boost::math::legendre_p(l, x) - boost::math::legendre_p(l - 1, x)) / (x * x - 1);
        };

        //i-th zero
        for (int i = 0; i < l / 2; i++) {
            //initial guess
            double x = std::cos(M_PI * (i + 1 - 0.25) / (l + 0.5));

            //Newton-Raphson iteration
            while (true) {
                double leg = boost::math::legendre_p(l, x);
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

    template<class T>
    void compute_integral_with_exp(
            const std::vector<double> &w,
            const std::vector<irlib::piecewise_polynomial<T>> &pp_func,
            Eigen::Tensor<std::complex<double>, 2> &Tnl
    );

    /// Construct piecewise polynomials representing Matsubara basis functions: exp(-i w_n tau) for n >= 0.
    /// For fermionic cases, w_n = (2*n+1)*pi/beta.
    /// For bosonci cases, w_n = (2*n)*pi/beta.
    /// Caution: when n is large, you need a very dense mesh. You are resposible for this.
    template<class T>
    void construct_matsubra_basis_functions_coeff(
            int n_min, int n_max,
            irlib::statistics::statistics_type s,
            const std::vector<double> &section_edges,
            int k,
            boost::multi_array<std::complex<T>, 3> &coeffs) {

        if (n_min < 0) {
            throw std::invalid_argument("n_min cannot be negative.");
        }
        if (n_min > n_max) {
            throw std::invalid_argument("n_min cannot be larger than n_max.");
        }

        const int N = section_edges.size() - 1;

        std::complex<double> z;
        coeffs.resize(boost::extents[n_max - n_min + 1][N][k + 1]);

        std::vector<double> pre_factor(k + 1);
        pre_factor[0] = 1.0;
        for (int j = 1; j < k + 1; ++j) {
            pre_factor[j] = pre_factor[j - 1] / j;
        }

        for (int n = n_min; n <= n_max; ++n) {
            if (s == irlib::statistics::FERMIONIC) {
                z = -std::complex<double>(0.0, n + 0.5) * M_PI;
            } else if (s == irlib::statistics::BOSONIC) {
                z = -std::complex<double>(0.0, n) * M_PI;
            }
            for (int section = 0; section < N; ++section) {
                const double x = section_edges[section];
                std::complex<T> exp0 = std::exp(z * (x + 1));
                std::complex<T> z_power = 1.0;
                for (int j = 0; j < k + 1; ++j) {
                    coeffs[n - n_min][section][j] = exp0 * z_power * pre_factor[j];
                    z_power *= z;
                }
            }
        }
    }

    /// Construct piecewise polynomials representing exponential functions: exp(i w_i x)
    template<class T>
    void construct_exp_functions_coeff(
            const std::vector<double> &w,
            const std::vector<double> &section_edges,
            int k,
            boost::multi_array<std::complex<T>, 3> &coeffs) {
        const int N = section_edges.size() - 1;

        std::complex<double> z;
        coeffs.resize(boost::extents[w.size()][N][k + 1]);

        std::vector<double> pre_factor(k + 1);
        pre_factor[0] = 1.0;
        for (int j = 1; j < k + 1; ++j) {
            pre_factor[j] = pre_factor[j - 1] / j;
        }

        for (int n = 0; n < w.size(); ++n) {
            auto z = std::complex<double>(0.0, w[n]);
            for (int section = 0; section < N; ++section) {
                const double x = section_edges[section];
                std::complex<T> exp0 = std::exp(z * (x + 1));
                std::complex<T> z_power = 1.0;
                for (int j = 0; j < k + 1; ++j) {
                    coeffs[n][section][j] = exp0 * z_power * pre_factor[j];
                    z_power *= z;
                }
            }
        }
    }

/**
 * Compute a transformation matrix from a give orthogonal basis set to Matsubara freq.
 * @tparam T  scalar type
 * @param n_min min index of Matsubara freq. index (>=0)
 * @param n_max max index of Matsubara freq. index (>=0)
 * @param statis Statistics (fermion or boson)
 * @param bf_src orthogonal basis functions. They must be piecewise polynomials of the same order.
 * @param Tnl  computed transformation matrix
 */
    template<class T>
    void compute_transformation_matrix_to_matsubara_impl(
            const std::vector<long> &n_vec,
            irlib::statistics::statistics_type statis,
            const std::vector<irlib::piecewise_polynomial<T>> &bf_src,
            Eigen::Tensor<std::complex<double>, 2> &Tnl
    ) {
        typedef std::complex<double> dcomplex;
        typedef irlib::piecewise_polynomial<std::complex<double> > pp_type;
        typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
        typedef Eigen::Tensor<std::complex<double>, 2> tensor_t;

        //if (n_min < 0) {
        //throw std::invalid_argument("n_min cannot be negative.");
        //}
        //if (n_min > n_max) {
        //throw std::invalid_argument("n_min cannot be larger than n_max.");
        //}
        for (int n = 0; n < n_vec.size(); ++n) {
            if (n_vec[n] < 0) {
                throw std::runtime_error("n_vec cannot be negative.");
            }
        }
        for (int n = 0; n < n_vec.size() - 1; ++n) {
            if (n_vec[n] > n_vec[n + 1]) {
                throw std::runtime_error("n_vec must be in ascending order.");
            }
        }

        std::vector<double> w(n_vec.size());

        for (int n = 0; n < n_vec.size(); ++n) {
            if (statis == irlib::statistics::FERMIONIC) {
                w[n] = (n_vec[n] + 0.5) * M_PI;
            } else if (statis == irlib::statistics::BOSONIC) {
                w[n] = n_vec[n] * M_PI;
            }
        }

        compute_integral_with_exp(w, bf_src, Tnl);

        std::vector<double> inv_norm(bf_src.size());
        for (int l = 0; l < bf_src.size(); ++l) {
            inv_norm[l] = 1. / std::sqrt(static_cast<double>(bf_src[l].overlap(bf_src[l])));
        }
        for (int n = 0; n < w.size(); ++n) {
            for (int l = 0; l < bf_src.size(); ++l) {
                Tnl(n, l) *= inv_norm[l] * std::sqrt(0.5);
            }
        }
    }

/**
 * Compute integral of exponential functions and given piecewise polynomials
 *           \int_{-1}^1 dx exp(i w_i (x+1)) p_j(x),
 *           where w_i are given real double objects and p_j are piecewise polynomials.
 *           The p_j(x) must be defined in the interval [-1,1].
 * @tparam T  scalar type of piecewise polynomials
 * @param w vector of w_i in ascending order
 * @param statis Statistics (fermion or boson)
 * @param p vector of piecewise polynomials
 * @param results  computed results
 */
    template<class T>
    void compute_integral_with_exp(
            const std::vector<double> &w,
            const std::vector<irlib::piecewise_polynomial<T>> &pp_func,
            Eigen::Tensor<std::complex<double>, 2> &Tnl
    ) {
        typedef std::complex<double> dcomplex;
        typedef irlib::piecewise_polynomial<std::complex<double> > pp_type;
        typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
        typedef Eigen::Tensor<std::complex<double>, 2> tensor_t;

        //order of polynomials used for representing exponential functions internally.
        const int k_iw = 16;
        const int k = pp_func[0].order();
        const int n_section = pp_func[0].num_sections();

        for (int l = 0; l < pp_func.size(); ++l) {
            if (k != pp_func[l].order()) {
                throw std::runtime_error(
                        "Error in compute_transformation_matrix_to_matsubara: basis functions must be pieacewise polynomials of the same order");
            }
            if (pp_func[l].section_edge(0) != -1 || pp_func[l].section_edge(n_section) != 1) {
                throw std::runtime_error("Piecewise polynomials must be defined on [-1,1]");
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

        boost::multi_array<std::complex<double>, 3> exp_coeffs(boost::extents[w.size()][n_section][k_iw + 1]);
        construct_exp_functions_coeff(w, pp_func[0].section_edges(), k_iw, exp_coeffs);

        matrix_t left_mid_matrix(n_iw, k + 1);
        matrix_t left_matrix(n_iw, k_iw + 1);
        matrix_t mid_matrix(k_iw + 1, k + 1);
        matrix_t right_matrix(k + 1, pp_func.size());
        matrix_t r(n_iw, pp_func.size());
        r.setZero();

        std::vector<double> dx_power(k + k_iw + 2);

        for (int s = 0; s < n_section; ++s) {
            double x0 = pp_func[0].section_edge(s);
            double x1 = pp_func[0].section_edge(s + 1);
            double dx = x1 - x0;
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
                        left_matrix(n, p) = exp_coeffs[n][s][p];
                    }
                }

                left_mid_matrix.block(0, 0, n_max_cs + 1, k + 1) =
                        left_matrix.block(0, 0, n_max_cs + 1, k_iw + 1) * mid_matrix;
            }

            //Otherwise, compute the overlap exactly
            for (int n = std::max(n_max_cs + 1, 0); n <= n_max; ++n) {
                std::complex<double> z = std::complex<double>(0.0, w[n]);

                dcomplex dx_z = dx * z;
                dcomplex dx_z2 = dx_z * dx_z;
                dcomplex dx_z3 = dx_z2 * dx_z;
                dcomplex inv_z = 1.0 / z;
                dcomplex inv_z2 = inv_z * inv_z;
                dcomplex inv_z3 = inv_z2 * inv_z;
                dcomplex inv_z4 = inv_z3 * inv_z;
                dcomplex exp = std::exp(dx * z);
                dcomplex exp0 = std::exp((x0 + 1.0) * z);

                left_mid_matrix(n, 0) = (-1.0 + exp) * inv_z * exp0;
                left_mid_matrix(n, 1) = ((dx_z - 1.0) * exp + 1.0) * inv_z2 * exp0;
                left_mid_matrix(n, 2) = ((dx_z2 - 2.0 * dx_z + 2.0) * exp - 2.0) * inv_z3 * exp0;
                left_mid_matrix(n, 3) = ((dx_z3 - 3.0 * dx_z2 + 6.0 * dx_z - 6.0) * exp + 6.0) * inv_z4 * exp0;
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
 * @param n_min min index of Matsubara freq. index (>=0)
 * @param n_max max index of Matsubara freq. index (>=0)
 * @param statis Statistics (fermion or boson)
 * @param bf_src orthogonal basis functions. They must be piecewise polynomials of the same order.
 * @param Tnl  computed transformation matrix
 */
    template<class T>
    void compute_transformation_matrix_to_matsubara(
            int n_min, int n_max,
            irlib::statistics::statistics_type statis,
            const std::vector<irlib::piecewise_polynomial<T>> &bf_src,
            Eigen::Tensor<std::complex<double>, 2> &Tnl
    ) {
        const int num_n = n_max - n_min + 1;
        const int batch_size = 500;
        Tnl = Eigen::Tensor<std::complex<double>, 2>(num_n, bf_src.size());
        Eigen::Tensor<std::complex<double>, 2> Tnl_batch(batch_size, bf_src.size());
        //TODO: use MPI
        //Split into batches to avoid using too much memory
        for (int ib = 0; ib < num_n / batch_size + 1; ++ib) {
            int n_min_batch = batch_size * ib;
            int n_max_batch = std::min(batch_size * (ib + 1) - 1, n_max);
            if (n_max_batch - n_min_batch < 0) {
                continue;
            }
            std::vector<long> n_vec;
            for (int n = n_min_batch; n <= n_max_batch; ++n) {
                n_vec.push_back(n);
            }
            compute_transformation_matrix_to_matsubara_impl(n_vec, statis, bf_src, Tnl_batch);
            for (int j = 0; j < bf_src.size(); ++j) {
                for (int n = n_min_batch; n <= n_max_batch; ++n) {
                    Tnl(n - n_min, j) = Tnl_batch(n - n_min_batch, j);
                }
            }
        }
    }

    /**
    * Compute a transformation matrix from a give orthogonal basis set to Matsubara freq.
    * @tparam T  scalar type
    * @param n indices of Matsubara frequqneices for which matrix elements will be computed (in strictly ascending order).
    *          The Matsubara basis functions look like exp(i PI * (n[i]+1/2)) for fermions, exp(i PI * n[i]) for bosons.
    * @param bf_src orthogonal basis functions. They must be piecewise polynomials of the same order.
    * @param Tnl  computed transformation matrix
    */
    template<class T>
    void compute_transformation_matrix_to_matsubara(
            const std::vector<long> &n,
            irlib::statistics::statistics_type statis,
            const std::vector<irlib::piecewise_polynomial<T>> &bf_src,
            Eigen::Tensor<std::complex<double>, 2> &Tnl
    ) {
        typedef std::complex<double> dcomplex;
        typedef irlib::piecewise_polynomial<std::complex<double> > pp_type;
        typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
        typedef Eigen::Tensor<std::complex<double>, 2> tensor_t;

        if (n.size() == 0) {
            return;
        }

        for (int i = 0; i < n.size() - 1; ++i) {
            if (n[i] > n[i + 1]) {
                throw std::runtime_error("n must be in strictly ascending order!");
            }
        }

        std::vector<double> w;
        if (statis == irlib::statistics::FERMIONIC) {
            std::transform(n.begin(), n.end(), std::back_inserter(w), [](double x) { return M_PI * (x + 0.5); });
        } else {
            std::transform(n.begin(), n.end(), std::back_inserter(w), [](double x) { return M_PI * x; });
        }

        compute_integral_with_exp(w, bf_src, Tnl);

        std::vector<double> inv_norm(bf_src.size());
        for (int l = 0; l < bf_src.size(); ++l) {
            inv_norm[l] = 1. / std::sqrt(static_cast<double>(bf_src[l].overlap(bf_src[l])));
        }
        for (int n = 0; n < w.size(); ++n) {
            for (int l = 0; l < bf_src.size(); ++l) {
                Tnl(n, l) *= inv_norm[l] * std::sqrt(0.5);
            }
        }
    }

    /**
    * Compute a transformation matrix (\bar{T}_{nl}) from a give orthogonal basis set to Matsubara freq.
    * @tparam T  scalar type
    * @param n indices of Matsubara frequqneices for which matrix elements will be computed (in strictly ascending order).
    *          The Matsubara basis functions look like exp(i PI * (n[i]/2) * (x+1)).
    * @param bf_src orthogonal basis functions. They must be piecewise polynomials of the same order.
    * @param Tnl  computed transformation matrix
    */
    template<class T>
    void compute_Tbar_ol(
            const std::vector<long> &o_vec,
            const std::vector<irlib::piecewise_polynomial<T>> &bf_src,
            Eigen::Tensor<std::complex<double>, 2> &Tbar_ol
    ) {
        typedef std::complex<double> dcomplex;
        typedef irlib::piecewise_polynomial<std::complex<double> > pp_type;
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

        std::vector<double> inv_norm(bf_src.size());
        for (int l = 0; l < bf_src.size(); ++l) {
            inv_norm[l] = 1. / std::sqrt(static_cast<double>(bf_src[l].overlap(bf_src[l])));
        }
        for (int n = 0; n < w.size(); ++n) {
            for (int l = 0; l < bf_src.size(); ++l) {
                Tbar_ol(n, l) *= inv_norm[l] * std::sqrt(0.5);
            }
        }
    }


    /// Compute overlap <left | right> with complex conjugate
    template<class T1, class T2>
    void compute_overlap(
            const std::vector<irlib::piecewise_polynomial<T1> > &left_vectors,
            const std::vector<irlib::piecewise_polynomial<T2> > &right_vectors,
            boost::multi_array<typename result_of_overlap<T1, T2>::value, 2> &results) {
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
        Eigen::Matrix<Tr, Eigen::Dynamic, Eigen::Dynamic> r(NL, NR);

        r.setZero();
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

            r += left_matrix * (mid_matrix * right_matrix);
        }

        results.resize(boost::extents[NL][NR]);
        for (int n = 0; n < NL; ++n) {
            for (int l = 0; l < NR; ++l) {
                results[n][l] = r(n, l);
            }
        }
    }


    /// Compute a transformation matrix from a src orthogonal basis set to a dst orthogonal basis set.
    /// The basis vectors are NOT necessarily normalized to 1.
    template<class T1, class T2>
    void compute_transformation_matrix(
            const std::vector<irlib::piecewise_polynomial<T1> > &dst_vectors,
            const std::vector<irlib::piecewise_polynomial<T2> > &src_vectors,
            boost::multi_array<typename result_of_overlap<T1, T2>::value, 2> &results) {
        compute_overlap(dst_vectors, src_vectors, results);

        std::vector<double> coeff1(dst_vectors.size());
        for (int l = 0; l < dst_vectors.size(); ++l) {
            coeff1[l] = 1.0 / std::sqrt(
                    static_cast<double>(
                            dst_vectors[l].overlap(dst_vectors[l])
                    )
            );
        }

        std::vector<double> coeff2(src_vectors.size());
        for (int l = 0; l < src_vectors.size(); ++l) {
            coeff2[l] = 1.0 / std::sqrt(
                    static_cast<double>(
                            src_vectors[l].overlap(src_vectors[l])
                    )
            );
        }

        for (int l1 = 0; l1 < dst_vectors.size(); ++l1) {
            for (int l2 = 0; l2 < src_vectors.size(); ++l2) {
                results[l1][l2] *= coeff1[l1] * coeff2[l2];
            }
        }
    };

    /**
     * Construct Piecewise polynomials approximately representing Legenre polynomials normalized to 1 on [-1,1]
     * @param Nl number of Legendre polynomials
     * @return Piecewise polynomials
     */
    inline std::vector<irlib::piecewise_polynomial<double>>
    construct_cubic_spline_normalized_legendre_polynomials(int Nl) {
        int Nl_max = 100;
        int M = 40;
        double eps = 1e-10;

        std::vector<double> nodes = compute_legendre_nodes(Nl_max);
        assert(Nl_max % 2 == 0);
        std::vector<double> positve_nodes;
        for (auto n: nodes) {
            if (n > 0.0) {
                positve_nodes.push_back(n);
            }
        }
        positve_nodes.push_back(0);
        positve_nodes.push_back(1);
        std::sort(positve_nodes.begin(), positve_nodes.end());

        std::vector<double> x_points;
        for (int i = 0; i < positve_nodes.size() - 1; ++i) {
            double dx = (positve_nodes[i + 1] - positve_nodes[i]) / M;
            for (int j = 0; j < M; ++j) {
                double x = positve_nodes[i] + dx * j;
                x_points.push_back(x);
                if (std::abs(x) > eps) {
                    x_points.push_back(-x);
                }
            }
        }
        x_points.push_back(1);
        x_points.push_back(-1);
        std::sort(x_points.begin(), x_points.end());

        std::vector<irlib::piecewise_polynomial<double>> results;
        std::vector<double> y_vals(x_points.size());
        for (int l = 0; l < Nl; ++l) {
            for (int j = 0; j < x_points.size(); ++j) {
                y_vals[j] = boost::math::legendre_p(l, x_points[j]) * std::sqrt(l + 0.5);
            }
            results.push_back(construct_piecewise_polynomial_cspline<double>(x_points, y_vals));
        }

        return results;
    }

    template<class Matrix, class Vector>
    void svd_square_matrix(Matrix &K, int n, Vector &S, Matrix &Vt, Matrix &U) {
        char jobu = 'S';
        char jobvt = 'S';
        int lda = n;
        int ldu = n;
        int ldvt = n;

        double *vt = Vt.data();
        double *u = U.data();
        double *s = S.data();

        double dummywork;
        int lwork = -1;
        int info = 0;

        double *A = K.data();
        std::vector<int> iwork(8 * n);

        //get optimal workspace
        dgesdd_(&jobu, &n, &n, A, &lda, s, u, &ldu, vt, &ldvt, &dummywork, &lwork, &iwork[0], &info);

        lwork = int(dummywork) + 32;
        Vector work(lwork);

        dgesdd_(&jobu, &n, &n, A, &lda, s, u, &ldu, vt, &ldvt, &work[0], &lwork, &iwork[0], &info);
        if (info != 0) {
            throw std::runtime_error("SVD failed to converge!");
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
    std::vector<double> compute_approximate_nodes_even_sector(const Kernel &knl, int N, double cutoff_singular_values) {
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
        Eigen::VectorXd svalues(N);
        matrix_t U(N, N), Vt(N, N);
        svd_square_matrix(K, N, svalues, Vt, U);

        //Count non-zero SV
        int dim = N;
        for (int i = 1; i < N; ++i) {
            if (std::abs(svalues(i) / svalues(0)) < cutoff_singular_values) {
                dim = i;
                break;
            }
        }

        //find nodes
        std::vector<double> nodes;
        for (int i = 0; i < N - 1; ++i) {
            if (U(i, dim - 1) * U(i + 1, dim - 1) < 0.0) {
                nodes.push_back(0.5 * (x_vec[i] + x_vec[i + 1]));
            }
        }

        if (nodes.size() != dim - 1) {
            throw std::runtime_error("The number of nodes is wrong.");
        }
    }
}
