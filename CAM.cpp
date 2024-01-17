#include <armadillo>
#include "CAM.h"

using namespace arma;


int main() {

      // Esempio di dati: 5 persone e 3 dati per persona, per ogni dato una matrice 2x2 (non vettore?)
      //arma::field<Mat> field_example(5, 3);  // Creerà un campo 3x3 di matrici vuote
      for (arma::uword i = 0; i < field_example.n_elem; i++) {
      field_example(i) = arma::randu(2, 2);  // Matrici 2x2 con valori casuali perche matrice e non vettore?
      }

      int n_j = field_example.n_cols; //numero di atomi, va cambiato quando non saranno tutti uguali (magari non saranno neanche tutti uguali)
      int J = 5; // numero persone
      int K = 2; // numero di DC
      int L = 3; // numero di OC
      int V = 2; // dimensione singola osservazione (vettore 2)
      int N = n_j; // numero di osservazioni per persona
      //vec N =    quando gli n_j non sono tutti uguali andrà passato un vettore (un n_j per ogni persona j)

      // Dimensioni da passare alla catena
      Dimensions dim = Dimensions {J, K, L, V, N};

      // Parametri per generare le Dirichlet
      double forma_beta = 5.0;
      double scala = 1.06;

      // Prior hyperparameters per le dirichlet
      vec alpha = arma::randg<vec>(dim.K, arma::distr_param(forma_beta, scala));
      vec beta = arma::randg<vec>(dim.L, arma::distr_param(forma_beta, scala));

      // Esempio di catena
      Chain catena_example = Chain(dim, alpha, beta, field_example);

      // Stampiamo le prior create
      catena_example.print();

      return 0;
}