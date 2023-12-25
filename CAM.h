#include <math.h>
#include "vector"
#include <armadillo>
#include <random>

using namespace std;
using namespace arma;


// TODO: Getter and Setter, spostare le variabili da public a private e cambiare di conseguenza il codice con i vari.get() e .set()














struct Dimensions{
    int J; //n of persons
    int K; //n of DC
    int L; //n of OC
    int V; //dimensions of one data
    int N; //numero di osservazioni per persona
}; //non  credo nj serva // si serve 


// funzione per generare campioni da una dirichlet di con parametro alpha (vettore)
vec generateDirichlet(const vec& alpha) {

    int k = alpha.n_elem;
    vec gammaSample(k);

    // Genera k variabili casuali dalla distribuzione gamma
    for (int i = 0; i < k; ++i) {
        gammaSample(i) = randg(1, distr_param(alpha(i), 1.0))(0);
    }

    // Normalizza il campione per ottenere una variabile casuale Dirichlet
    return gammaSample / sum(gammaSample);
}



class Chain{
    // ANDREBBERO MESSE TUTTE LE VARIABILI NEL PRIOR E CREATI GETTERS E SETTERS

    /*
private:
    field<mat> data;
    Dimensions dim;
    vec pi; // (K x 1) pesi per scegliere DC
    vec alpha; // (K x 1) K:= numero di DC
    mat w; // (L x K) scelto K, pesi per scegliere OC
    vec beta; // (L x 1) L := numero di OC
    Col<int> S; // (J x 1) assegna un DC ad ogni persona (J persone)
    Mat<int> M; // (J x max(n_j)) assegna per ogni persona un OC ad ogni atomo di quella persona (J persone e n_j atomo per persona j)
    */


public:
    field<mat> data; //input
    Dimensions dim; //input
    vec pi; // (K x 1) pesi per scegliere DC
    vec alpha; // (K x 1) K:= numero di DC (passato in input)
    mat w; // (L x K) scelto K, pesi per scegliere OC
    vec beta; // (L x 1) L := numero di OC (passato in input)
    arma::Col<int> S; // (J x 1) assegna un DC ad ogni persona (J persone)
    Mat<int> M; // (J x max(n_j)) assegna per ogni persona un OC ad ogni atomo di quella persona (J persone e n_j atomo per persona j)

    // Costruttore 


    Chain(const Dimensions& input_dim, const vec& input_alpha, const vec& input_beta, const arma::field<mat>& input_data) {

        dim = input_dim;
        alpha = input_alpha;
        beta = input_beta;
        data = input_data;

        // Generazione dei pesi pi e w
        pi = generateDirichlet(alpha); // genera i pesi per DC dalla Dirichlet
        for (int k = 0; k < dim.K; k++) {
            vec w_k = generateDirichlet(beta); //genera un vettore dalla dirichlet per ogni k
            w = join_rows(w, w_k); // costruisce la matrice dei pesi aggiungendo ogni colonna
        }

        // Generazione delle variabili categoriche S e M
        std::default_random_engine generatore_random;

        discrete_distribution<int> Cat(pi.begin(), pi.end()); 

        S.set_size(dim.J);
        M.set_size(dim.J, dim.N);
        
        for (int j = 0; j < dim.J; j++) {
            S(j) = Cat(generatore_random); // genera S (vettore lungo J persone) con ogni elemento preso dalla categorica con pesi pi
            discrete_distribution<int> Cat_w(w.col(S(j)).begin(), w.col(S(j)).end()); //avendo deciso S per la persona j, costruisco la distribuzione categorica con i pesi associati a w_S(j)
            for (int i = 0; i < dim.N; i++) {
                M(j,i) = Cat_w(generatore_random); // genera M(matrice J x N) per ogni persona (j), genera N valori da categoriche seguendo la distribuzione Cat_w
            }
        }
        
    }


    // Stampa
    void print(void) {
        cout << "pi:\n" << pi << endl;
        cout << "w:\n" << w << endl;
        cout << "S:\n" << S << endl;
        cout << "M:\n" << M << endl;
    }



};
