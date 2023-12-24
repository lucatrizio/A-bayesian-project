#include <armadillo>
#include "CAM.h"

using namespace arma;


int main() {
  arma::field<Mat> field_example(5, 3);  // Creerà un campo 3x3 di matrici vuote
  for (arma::uword i = 0; i < field_example.n_elem; i++) {
        field_example(i) = arma::randu(2, 2);  // Matrici 2x2 con valori casuali
  }
  int n_j = field_example.n_cols;
  int J = 5; // numero persone
  int K = 2; // numero di DC
  int L = 3; // numero di OC
  int V = 2; // dimensioni dati
  //vec N = arma::ones<arma::vec>(J) * n_j;  // esempio, poi n_j magari non è uguale per tutti
  int N = n_j;
  Dimensions dim = Dimensions {J, K, L, V, N};
  double forma_beta = 5.0;
  double scala = 1.06;
  // Generazione di un vettore di alpha
  vec alpha = arma::randg<vec>(dim.K, arma::distr_param(forma_beta, scala));
  vec beta = arma::randg<vec>(dim.L, arma::distr_param(forma_beta, scala));
  Chain catena_example = Chain(dim, alpha, beta, field_example);
  catena_example.print();

  return 0;
  }