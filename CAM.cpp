#include <armadillo>
#include "CAM.h"
#include <RInside.h>
#include <Rcpp.h>

using namespace arma;


int main() {

      // Esempio di dati: 5 persone e 3 dati per persona, per ogni dato una matrice 2x2 (non vettore?)
      size_t n = 5; // Number of people
      size_t observations[] = {3,3,3,3,3}; // Number of observations for each person
      size_t v = 2; // Number of observed parameters for each observation
      Data field_example(n,observations,v);  // Creerà un campo 3x3 di matrici vuote
      /*for (arma::uword i = 0; i < field_example.n_elem; i++) {
      field_example(i) = arma::randu(2, 2);  // Matrici 2x2 con valori casuali perche matrice e non vettore?
      }
      */

     for (size_t i = 0; i < n; i++)
     {
            for (size_t j = 0; j < observations[i]; j++)
            {
                  for (size_t k = 0; k < v; k++)
                  {
                        field_example.set(i,j,k,arma::randu());
                  }
                  
            }
            
     }
     

      //int n_j = field_example.n_cols; //numero di atomi, va cambiato quando non saranno tutti uguali (magari non saranno neanche tutti uguali)
      int J = 5; // numero persone
      int K = 2; // numero di DC
      int L = 3; // numero di OC
      int V = 2; // dimensione singola osservazione (vettore 2)
      int N = 3; // numero di osservazioni per persona
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
      field_example.print();
      catena_example.print();

      return 0;
}