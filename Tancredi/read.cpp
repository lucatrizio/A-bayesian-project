#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <armadillo>

using namespace std;

arma::mat main() {
    // Apri il file CSV in modalità lettura
    ifstream file("test_data.csv");

    // Verifica se il file è stato aperto correttamente
    if (!file.is_open()) {
        cerr << "Errore nell'apertura del file CSV." << endl;
        return 1;
    }

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
