#include <armadillo>
#include "Chain.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace arma;
using namespace std;


int main() {

      // Esempio di dati: 5 persone e 3 dati per persona, per ogni dato una matrice 2x2 (non vettore?)
      size_t J = 15; // Number of people
      size_t observations[] = {3,4,5,3,4,5,3,4,5,3,4,5,3,4,5}; // Number of observations for each person
      size_t v = 3; // Number of observed parameters for each observation
      Data field_example(J,observations,v);  // Creerà un campo 3x3 di matrici vuote
      /*for (arma::uword i = 0; i < field_example.n_elem; i++) {
      field_example(i) = arma::randu(2, 2);  // Matrici 2x2 con valori casuali perche matrice e non vettore?
      }
      */
     mat csv = read();

     for (size_t j = 0; j < J; j++)
     {
            for (size_t i = 0; i < observations[j]; i++) { 
                  vec vet(v, arma::fill::randn);
                  if (j%2 == 0) {
                        vet = vet + 5;
                  }
                  if (j%3 == 0) {
                        vet = vet + 15;
                  }
                  field_example.set_vec(j,i, vet);
            }    
     }
     field_example.print();


      //int n_j = field_example.n_cols; //numero di atomi, va cambiato quando non saranno tutti uguali (magari non saranno neanche tutti uguali)
      //size_t J = 5; // numero persone
      size_t K = 5; // numero di DC
      size_t L = 3; // numero di OC
      //size_t v = 2; // dimensione singola osservazione (vettore 2)
      size_t max_N = 5; // numero di osservazioni per persona
      size_t* N = observations;   // quando gli n_j non sono tutti uguali andrà passato un vettore (un n_j per ogni persona j)

      // Dimensioni da passare alla catena
      Dimensions dim = Dimensions {J, K, L, v, observations, max_N};

      // Parametri per generare le Dirichlet
      double forma_beta = 1.0;
      double scala = 1.06;

      // Prior hyperparameters per le dirichlet
      vec alpha = arma::randg<vec>(dim.K, arma::distr_param(forma_beta, scala));
      vec beta_col = arma::randg<vec>(dim.L, arma::distr_param(forma_beta, scala));
      mat beta = arma::repmat(beta_col, 1, K);
      // Esempio di catena
      Chain catena_example = Chain(dim, alpha, beta, field_example);
      size_t B = 100;
      catena_example.print();
      for (size_t b = 0; b < B; ++b) {
            catena_example.chain_step();  
      }
      catena_example.print();
      // Stampiamo le prior create
      return 0;
}



#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <armadillo>

using namespace std;

arma::mat read() {
    // Apri il file CSV in modalità lettura
    ifstream file("test_data.csv");

    // Dichiarazione di variabili per la lettura del file
    string line, token;
    vector<vector<string>> data; // Struttura dati per memorizzare i dati CSV

    // Leggi il file riga per riga
    while (getline(file, line)) {
        stringstream ss(line);
        vector<string> row; // Vettore per memorizzare i dati di ogni riga

        // Estrai i token dalla riga utilizzando una virgola come delimitatore
        while (getline(ss, token, ',')) {
            row.push_back(token);
        }

        // Aggiungi la riga al vettore dei dati
        data.push_back(row);
    }

    // Chiudi il file dopo aver letto i dati
    file.close();

    arma::mat armaMatrix(data.size(), data[0].size());

    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[0].size(); ++j) {
            armaMatrix(i, j) = stod(data[i][j]);
        }
    }

    return armaMatrix;
}
