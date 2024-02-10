#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

int main() {
    // Apri il file CSV in modalità lettura
    ifstream file("data.CSV");

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

    // Ora puoi utilizzare il vettore "data" per accedere ai dati CSV

    // Esempio di come stampare i dati
    for (const auto& row : data) {
        for (const auto& value : row) {
            cout << value << " ";
        }
        cout << endl;
    }

    return 0;
}
