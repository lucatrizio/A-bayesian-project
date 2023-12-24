#include <math.h>
#include "vector"
#include <armadillo>
#include <random>

using namespace std;
using namespace arma;

struct Dimensions{
    int J; //n of persons
    int K; //n of DC
    int L; //n of OC
    int V; //dimensions of one data
    int N;
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
    field<mat> data;
    Dimensions dim;
    vec pi; // (K x 1) pesi per scegliere DC
    vec alpha; // (K x 1) K:= numero di DC
    mat w; // (L x K) scelto K, pesi per scegliere OC
    vec beta; // (L x 1) L := numero di OC
    arma::Col<int> S; // (J x 1) assegna un DC ad ogni persona (J persone)
    Mat<int> M; // (J x max(n_j)) assegna per ogni persona un OC ad ogni atomo di quella persona (J persone e n_j atomo per persona j)
    Chain(const Dimensions& input_dim, const vec& input_alpha, const vec& input_beta, const arma::field<mat>& input_data) {
        dim = input_dim;
        alpha = input_alpha;
        beta = input_beta;
        data = input_data;
        pi = generateDirichlet(alpha); // genera i pesi per DC dalla Dirichlet
        for (int k = 0; k < dim.K; k++) {
            vec w_k = generateDirichlet(beta);
            w = join_rows(w, w_k); // genera i pesi per OC dalla Dirichlet
        }
        std::default_random_engine generatore_random;
        discrete_distribution<int> Cat(pi.begin(), pi.end()); 
        S.set_size(dim.J);
        M.set_size(dim.J, dim.N);
        for (int j = 0; j < dim.J; j++) {
            S(j) = Cat(generatore_random); // genera S (vettore lungo J persone) con ogni elemento preso dalla categorica con pesi pi
            discrete_distribution<int> Cat_w(w.col(S(j)).begin(), w.col(S(j)).end()); 
            for (int i = 0; i < dim.N; i++) {
                M(j,i) = Cat_w(generatore_random);
            }
        }
        
    }

    void print(void) {
        cout << "pi:\n" << pi << endl;
        cout << "w:\n" << w << endl;
        cout << "S:\n" << S << endl;
        cout << "M:\n" << M << endl;
    }
};
