#pragma once

#include <complex>
#include <memory>
#include <utility>

#include "common.hpp"

namespace irlib {
    /**
     * Abstract class representing an analytical continuation kernel
     */
    template<typename T>
    class kernel {
    public:
        typedef T mp_type;

        virtual ~kernel() {};

        /// return the value of the kernel for given x and y in the [-1,1] interval.
        virtual T operator()(T x, T y) const = 0;

        /// return statistics
        virtual irlib::statistics::statistics_type get_statistics() const = 0;

        /// return lambda
        virtual double Lambda() const = 0;

#ifndef SWIG

        /// return a reference to a copy
        virtual std::shared_ptr<kernel> clone() const = 0;

#endif
    };

#ifdef SWIG
    %template(real_kernel) kernel<mpfr::mpreal>;
#endif

    /**
     * Fermionic kernel
     */
    class fermionic_kernel : public kernel<mpfr::mpreal> {
    public:
        fermionic_kernel(double Lambda) : Lambda_(Lambda) {}

        virtual ~fermionic_kernel() {};

        mpfr::mpreal operator()(mpfr::mpreal x, mpfr::mpreal y) const {
            mpfr::mpreal half_Lambda = mpfr::mpreal("0.5") * mpfr::mpreal(Lambda_);

            const double limit = 200.0;
            if (Lambda_ * y > limit) {
                return mpfr::exp(-half_Lambda * x * y - half_Lambda * y);
            } else if (Lambda_ * y < -limit) {
                return mpfr::exp(-half_Lambda * x * y + half_Lambda * y);
            } else {
                return mpfr::exp(-half_Lambda * x * y) / (2 * mpfr::cosh(half_Lambda * y));
            }
        }

        irlib::statistics::statistics_type get_statistics() const {
            return irlib::statistics::FERMIONIC;
        }

        double Lambda() const {
            return Lambda_;
        }

#ifndef SWIG

        std::shared_ptr<kernel> clone() const {
            return std::shared_ptr<kernel>(new fermionic_kernel(Lambda_));
        }

#endif

    private:
        double Lambda_;
    };

    /**
     * Bosonic kernel
     */
    class bosonic_kernel : public kernel<mpfr::mpreal> {
    public:
        bosonic_kernel(double Lambda) : Lambda_(Lambda) {}

        virtual ~bosonic_kernel() {};

        mpfr::mpreal operator()(mpfr::mpreal x, mpfr::mpreal y) const {
            const double limit = 200.0;
            mpfr::mpreal half_Lambda = mpfr::mpreal("0.5") * mpfr::mpreal(Lambda_);

            if (mpfr::abs(Lambda_ * y) < 1e-30) {
                return mpfr::exp(-half_Lambda * x * y) / Lambda_;
            } else if (Lambda_ * y > limit) {
                return y * mpfr::exp(-half_Lambda * x * y - half_Lambda * y);
            } else if (Lambda_ * y < -limit) {
                return -y * mpfr::exp(-half_Lambda * x * y + half_Lambda * y);
            } else {
                return y * mpfr::exp(-half_Lambda * x * y) / (2 * mpfr::sinh(half_Lambda * y));
            }
        }

        irlib::statistics::statistics_type get_statistics() const {
            return irlib::statistics::BOSONIC;
        }

        double Lambda() const {
            return Lambda_;
        }

#ifndef SWIG

        std::shared_ptr<kernel> clone() const {
            return std::shared_ptr<kernel>(new bosonic_kernel(Lambda_));
        }

#endif

    private:
        double Lambda_;
    };

    namespace detail {

        template<typename mp_type>
        std::vector<std::pair<mp_type,mp_type> >
        composite_gauss_legendre_nodes(
                const std::vector<mp_type>& section_edges,
                const std::vector<std::pair<mp_type,mp_type> >& nodes
        ) {
            int num_sec = section_edges.size() - 1;
            int num_local_nodes = nodes.size();

            std::vector<std::pair<mp_type,mp_type> > all_nodes(num_sec*num_local_nodes);
            for (int s=0; s<num_sec; ++s) {
                auto a = section_edges[s];
                auto b = section_edges[s+1];
                for (int n=0; n<num_local_nodes; ++n) {
                    mp_type x = a + ((b-a)/mp_type(2)) * (nodes[n].first+mp_type(1));
                    mp_type w = ((b-a)/mp_type(2)) * nodes[n].second;
                    all_nodes[s * num_local_nodes + n] = std::make_pair(x, w);
                }
            }
            return all_nodes;
        };
    }


    /**
     * Compute Matrix representation of a given Kernel
     * @tparam mp_type
     * @tparam K
     * @param kernel
     * @param section_edges_x
     * @param section_edges_y
     * @param num_local_nodes
     * @param Nl
     * @return Matrix representation
     */
    template<typename mp_type, typename K>
    Eigen::Matrix<mp_type, Eigen::Dynamic, Eigen::Dynamic>
    matrix_rep(const K& kernel,
               const std::vector<mp_type>& section_edges_x,
               const std::vector<mp_type>& section_edges_y,
               int num_local_nodes,
               int Nl) {

        using matrix_type = Eigen::Matrix<mp_type, Eigen::Dynamic, Eigen::Dynamic>;

        int num_sec = section_edges_x.size()-1;

        //mp_type dx = section_edges_x.back() - section_edges_x[0];
        //mp_type dy = section_edges_y.back() - section_edges_y[0];

        // nodes for Gauss-Legendre integration
        std::vector<std::pair<mp_type ,mp_type >> nodes = detail::gauss_legendre_nodes<mp_type>(num_local_nodes);
        auto nodes_x = detail::composite_gauss_legendre_nodes(section_edges_x, nodes);
        auto nodes_y = detail::composite_gauss_legendre_nodes(section_edges_y, nodes);

        std::vector<matrix_type> phi_x(num_sec);
        std::vector<matrix_type> phi_y(num_sec);
        for(int s=0; s<num_sec; ++s) {
            phi_x[s] = matrix_type(Nl, num_local_nodes);
            phi_y[s] = matrix_type(Nl, num_local_nodes);
            for (int n=0; n<num_local_nodes; ++n) {
                for (int l=0; l<Nl; ++l) {
                    auto leg_val = normalized_legendre_p(l, nodes[n].first);
                    phi_x[s](l, n) = mpfr::sqrt(mp_type(2)/(section_edges_x[s+1]-section_edges_x[s])) * leg_val * nodes_x[s*num_local_nodes + n].second;
                    phi_y[s](l, n) = mpfr::sqrt(mp_type(2)/(section_edges_y[s+1]-section_edges_y[s])) * leg_val * nodes_y[s*num_local_nodes + n].second;
                }
            }
        }

        matrix_type K_mat(num_sec*Nl, num_sec*Nl);
        for (int s2 = 0; s2 < num_sec; ++s2) {
            for (int s = 0; s < num_sec; ++s) {

                matrix_type K_nn(num_local_nodes, num_local_nodes);
                for (int n=0; n<num_local_nodes; ++n) {
                    for (int n2 = 0; n2 < num_local_nodes; ++n2) {
                        K_nn(n, n2) = kernel(nodes_x[s*num_local_nodes+n].first, nodes_y[s2*num_local_nodes+n2].first);
                    }
                }

                // phi_x(l, n) * K_nn(n, n2) * phi_y(l2, n2)^T
                matrix_type r = phi_x[s] * K_nn * phi_y[s2].transpose();

                for (int l2=0; l2<Nl; ++l2) {
                    for (int l = 0; l < Nl; ++l) {
                        K_mat(Nl*s+l, Nl*s2+l2) = r(l, l2);
                    }
                }
            }
        }

        return K_mat;
    }
}
