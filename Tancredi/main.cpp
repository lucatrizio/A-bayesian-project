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

    ifstream file("nuovo_test_data_pulito.csv");

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



    size_t J = 10; // Number of people
    size_t observations[] = {15,15,15,15,15,15,15,15,15,15}; // Number of observations for each person
    size_t v = 3; // Number of observed parameters for each observation
    Data field_example(J,observations,v);  // Creerà un campo 3x3 di matrici vuote

    arma::mat armaMatrix(150, 3, arma::fill::zeros);


    for (size_t i = 0; i < d.size(); i++) {
        for (size_t j = 0; j < d[0].size(); j++) {
            double val = stod(d[i][j]);
            armaMatrix(i, j) = val;
        }
    }
    //armaMatrix.print();

    size_t u = 0;
    for (size_t j = 0; j < J; j++) {
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


    size_t K = 10; // numero di DC
    size_t L = 15 ; // numero di OC // PROVA CON L >> K (3/2)
    size_t max_N = 15; // numero di osservazioni per persona
    size_t* N = observations;   // quando gli n_j non sono tutti uguali andrà passato un vettore (un n_j per ogni persona j)

    // Dimensioni da passare alla catena
    Dimensions dim = Dimensions {J, K, L, v, observations, max_N};

    // Parametri per generare le Dirichlet


    // Prior hyperparameters per le dirichlet
    vec alpha;
    alpha.ones(dim.K);
    alpha = alpha*0.01;

    vec beta_col;
    beta_col.ones(dim.L);
    beta_col = beta_col * 0.01;
    mat beta = arma::repmat(beta_col, 1, dim.K);


    vec mu = arma::mean(armaMatrix, 0).t();
    double lambda = 0.01;
    int nu = 3 + 2;
    mat scale_matrix = arma::cov(armaMatrix);


    Chain catena_example = Chain(dim, alpha, beta, field_example, mu, lambda, nu, scale_matrix);


    size_t B = 3000;
    catena_example.print();

    for (size_t b = 0; b < B; ++b) {
        cout << "=== " << b <<" ===" <<endl;
        catena_example.chain_step();  
    }

    catena_example.print();
    vec results = {1,2,2,1,2,1,1,2,1,2};
    cout << "results" << endl;
    results.print();
    return 0;
}



