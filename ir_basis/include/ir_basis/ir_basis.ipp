#include "ir_basis.hpp"

namespace ir {
  namespace detail {
    template<typename T>
    inline std::vector<T> linspace(T minval, T maxval, int N) {
      std::vector<T> r(N);
      for (int i = 0; i < N; ++i) {
        r[i] = i * (maxval - minval) / (N - 1) + minval;
      }
      return r;
    }

    template<class Matrix, class Vector>
    void svd_square_matrix(Matrix &K, int n, Vector &S, Matrix &Vt, Matrix &U){
      char jobu = 'S';
      char jobvt = 'S';
      int lda = n;
      int ldu = n;
      int ldvt = n;

      double *vt = Vt.data();
      double *u = U.data();
      double *s = S.data();

      double dummywork;
      int lwork=-1;
      int info=0;

      double *A = K.data();

      //get optimal workspace
      dgesvd_(&jobu,&jobvt,&n,&n,A,&lda,s,u,&ldu,vt,&ldvt,&dummywork,&lwork,&info);

      lwork=int(dummywork)+32;
      Vector work(lwork);

      dgesvd_(&jobu, &jobvt, &n, &n, A, &lda, s, u, &ldu, vt, &ldvt, &work[0], &lwork, &info);
      if(info!=0) {
        throw std::runtime_error("SVD failed to converge!");
      }
    }
  }

  template<typename Kernel>
  void do_svd(double Lambda, int parity, int N, double cutoff_singular_values,
              std::vector<double>& singular_values,
              std::vector<piecewise_polynomial<double,3> >& basis_functions
  ) {
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

    double de_cutoff = 2.5;

    //DE mesh for x
    std::vector<double> tx_vec = detail::linspace<double>(0.0, de_cutoff, N);
    std::vector<double> weight_x(N), x_vec(N);
    for (int i = 0; i < N; ++i) {
      x_vec[i] = std::tanh(0.5*M_PI*std::sinh(tx_vec[i]));
      //sqrt of the weight of DE formula
      weight_x[i] = std::sqrt(0.5*M_PI*std::cosh(tx_vec[i]))/std::cosh(0.5*M_PI*std::sinh(tx_vec[i]));
    }

    //DE mesh for y
    std::vector<double> ty_vec = detail::linspace<double>(-de_cutoff, 0.0, N);
    std::vector<double> y_vec(N), weight_y(N);
    for (int i = 0; i < N; ++i) {
      y_vec[i] = std::tanh(0.5*M_PI*std::sinh(ty_vec[i])) + 1.0;
      //sqrt of the weight of DE formula
      weight_y[i] = std::sqrt(0.5*M_PI*std::cosh(ty_vec[i]))/std::cosh(0.5*M_PI*std::sinh(ty_vec[i]));
    }

    Kernel k_obj(Lambda);
    matrix_t K(N, N);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        K(i, j) = weight_x[i] * (k_obj(x_vec[i], y_vec[j]) + parity * k_obj(x_vec[i], -y_vec[j])) * weight_y[j];
      }
    }

    //Perform SVD
    Eigen::VectorXd svalues(N);
    matrix_t U(N,N), Vt(N,N);
    detail::svd_square_matrix(K, N, svalues, Vt, U);

    //Count non-zero SV
    int dim = 0;
    for (int i = 1; i < N; ++i) {
      if (std::abs(svalues(i)/svalues(0)) < cutoff_singular_values) {
        dim = i;
        break;
      }
    }

    //Rescale U and V
    U.conservativeResize(N, dim);
    for (int l = 0; l < dim; ++l) {
      for (int i = 0; i < N; ++i) {
        U(i,l) /= weight_x[i];
      }
      if (U(N-1, l) < 0.0) {
        U.col(l) *= -1;
      }
    }

    singular_values.resize(dim);
    for (int l = 0; l < dim; ++l) {
      singular_values[l] = svalues(l);
    }

    //cubic spline interpolation
    const int n_points = 2*N + 1;
    const int n_section = n_points + 1;
    std::vector<double> x_array(n_points), y_array(n_points);

    //set up x values
    for (int itau = 0; itau < N; ++itau) {
      x_array[-itau + n_points/2] = -x_vec[itau];
      x_array[itau + n_points/2] = x_vec[itau];
    }
    x_array.front() = -1.0;
    x_array.back() = 1.0;

    // spline interpolation
    for (int l = 0; l < dim; ++l) {
      //set up y values
      for (int itau = 0; itau < N; ++itau) {
        y_array[-itau + n_points/2] = parity * U(itau, l);
        y_array[itau + n_points/2] = U(itau, l);
      }
      if (parity == -1) {
        y_array[n_points/2] = 0.0;
      }
      y_array.front() = parity * U(N-1, l);
      y_array.back() = U(N-1, l);

      basis_functions.push_back(construct_piecewise_polynomial_cspline<double>(x_array, y_array));
    }

    orthonormalize(basis_functions);
  }

  template<typename Scalar, typename Kernel>
  Basis<Scalar,Kernel>::Basis(double Lambda, int N, double cutoff) {

    std::vector<double> even_svalues, odd_svalues, svalues;
    std::vector<pp_type> even_basis_functions, odd_basis_functions;

    do_svd<Kernel>(Lambda,  1, N, cutoff, even_svalues, even_basis_functions);
    do_svd<Kernel>(Lambda, -1, N, cutoff, odd_svalues, odd_basis_functions);

    //Merge
    basis_functions_.resize(0);
    assert(even_basis_functions.size() == even_svalues.size());
    assert(odd_basis_functions.size() == odd_svalues.size());
    for (int pair = 0; pair < std::max(even_svalues.size(), odd_svalues.size()); ++pair) {
      if (pair < even_svalues.size()) {
        svalues.push_back(even_svalues[pair]);
        basis_functions_.push_back(even_basis_functions[pair]);
      }
      if (pair < odd_svalues.size()) {
        svalues.push_back(odd_svalues[pair]);
        basis_functions_.push_back(odd_basis_functions[pair]);
      }
    }

    assert(even_svalues.size() + odd_svalues.size() == svalues.size());

    //Check
    for (int i = 0; i < svalues.size() - 1; ++i) {
      if (svalues[i] < svalues[i+1]) {
        //FIXME: SHOULD NOT THROW IN A CONSTRUCTOR
        throw std::runtime_error("Even and odd basis functions do not appear alternately.");
      }
    }
  };

  template<typename Scalar, typename Kernel>
  void
  Basis<Scalar,Kernel>::value(double x, std::vector<double> &val) const {
    assert(val.size() >= basis_functions_.size());
    assert(x >= -1.00001 && x <= 1.00001);

    const int dim = basis_functions_.size();

    if (dim > val.size()) {
      val.resize(dim);
    }
    const int section = basis_functions_[0].find_section(x);
    for (int l = 0; l < dim; l++) {
      val[l] = basis_functions_[l].compute_value(x, section);
    }
  }
}

