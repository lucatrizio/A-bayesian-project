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

         ifstream file("test_data.csv");

    // Dichiarazione di variabili per la lettura del file
    string line, token;
    vector<vector<string>> d; // Struttura dati per memorizzare i dati CSV

    // Leggi il file riga per riga
    while (getline(file, line)) {
        stringstream ss(line);
        vector<string> row; // Vettore per memorizzare i dati di ogni riga

        // Estrai i token dalla riga utilizzando una virgola come delimitatore
        while (getline(ss, token, ',')) {
            row.push_back(token);
        }

        // Aggiungi la riga al vettore dei dati
        d.push_back(row);
    }


    // Chiudi il file dopo aver letto i dati
    file.close();

    arma::mat armaMatrix(60, 3, arma::fill::zeros);

    for (size_t i = 0; i < d.size(); i++) {
        for (size_t j = 0; j < d[0].size(); j++) {
            double val = stod(d[i][j]);
            armaMatrix(i, j) = val;
        }
    }
     size_t u = 0;
     for (size_t j = 0; j < J; j++)
     {
            for (size_t i = 0; i < observations[j]; i++) { 
                vec vet(3, arma::fill::zeros);
                for (size_t t = 0; t < 3; t++) {
                    vet(t) = armaMatrix(u,t);
                }
                field_example.set_vec(j,i, vet);
                u++;
            }    
     }
     // field_example.print();

      // SE NON 
      //int n_j = field_example.n_cols; //numero di atomi, va cambiato quando non saranno tutti uguali (magari non saranno neanche tutti uguali)
      //size_t J = 5; // numero persone
      size_t K = 5; // numero di DC
      size_t L = 8; // numero di OC // PROVA CON L >> K (3/2)
      //size_t v = 2; // dimensione singola osservazione (vettore 2)
      size_t max_N = 5; // numero di osservazioni per persona
      size_t* N = observations;   // quando gli n_j non sono tutti uguali andrà passato un vettore (un n_j per ogni persona j)

      // Dimensioni da passare alla catena
      Dimensions dim = Dimensions {J, K, L, v, observations, max_N};

      // Parametri per generare le Dirichlet
      double forma_beta = .5;
      double scala = 3.06;

      // Prior hyperparameters per le dirichlet
      vec alpha = arma::randg<vec>(dim.K, arma::distr_param(forma_beta, scala));
      vec beta_col = arma::randg<vec>(dim.L, arma::distr_param(forma_beta, scala));
      mat beta = arma::repmat(beta_col, 1, K);
      // Esempio di catena
      Chain catena_example = Chain(dim, alpha, beta, field_example);
      size_t B = 1;
      //catena_example.print();
      //std::cout << "prima " <<std::endl; 
      for (size_t b = 0; b < B; ++b) {
            std::cout << b << std::endl;
            catena_example.chain_step();  
      }
      //std::cout << "dopo " <<std::endl; 
      //catena_example.print();
      // Stampiamo le prior create
      return 0;
}



