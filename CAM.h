#include <math.h>
#include "vector"
#include <armadillo>
#include <random>
#include <iostream>

using namespace std;
using namespace arma;


class Theta{
private:
    struct Parameters
    {
        float*mean;
        float**covariance;
    };

    Parameters*theta; //vector of parameters(mean and coavariance)
    size_t size_L, size_v; //Number of L observational clusters (normal distributions each with a mean and a covariance)
    
public:
    // Constructor
    Theta(size_t L, size_t v) : size_L(L), size_v(v)
    {
        theta = new Parameters[L];

        for (size_t i = 0; i < L; i++)
        {
            theta[i].mean = new float[v];
            theta[i].covariance = new float *[v];

            for (size_t j = 0; j < count; j++)
            {
                theta[i].covariance[j] = new float[v];
            }
            
        }
        
    }

    // Destructor
    ~Theta()
    {
        for (size_t i = 0; i < size_L; ++i)
        {
            delete[] theta[i].mean;

            for (size_t j = 0; j < size_v; j++)
            {
                delete[] theta[i].covariance[j];
            }
            delete[] theta[i].covariance;
        }

        delete[] theta;
    }

    void print(){
        for (size_t i = 0; i < size_L; i++)
        {
            std::cout << "mean" << i << ":(";
            for (size_t j = 0; j < size_v; j++)
            {
                std::cout << theta[i].mean[j] << ",";
            }
            std::cout << ")  covariance" << i << ":(";

            for (size_t j = 0; j < size_v; j++)
            {
                std::cout << "[";
                for (size_t k = 0; k < size_v; k++)
                {
                    std::cout << theta[i].covariance[j][k] << ",";
                }
                std::cout << "] ";
                
            }

            std::cout << ")";
            std::cout << std::endl;
            
        } 
    }

    void set(size_t s, float *mean, float **covariance)
    {

        for (size_t i = 0; i < count; i++)
        {
            theta[s].mean[i] = mean[i];

            for (size_t j = 0; j < count; j++)
            {
                theta[s].coavariance[i][j] = covariance[i][j];
            }
        }
    }

    float *getMean(size_t i)
    {
        return theta[i].mean;
    }

    float **getCovariate(size_t i)
    {
        return theta[i].covariance;
    }

}

class Data {
private:
    float***data;
    size_t n, //number of people
    *observations, //number of observations for each person
    v; //number of observed parameters for each observation

public:
    // Constructor
    Data(size_t n, size_t *observations, size_t v) : n(n), observations(observations), v(v)
    {

        data = new float **[n];

        for (size_t i = 0; i < n; ++i)
        {
            data[i] = new float *[observations[i]];

            for (size_t j = 0; j < observations[i]; ++j)
            {
                data[i][j] = new float[v];
            }
        }
    }

    // Destructor
    ~Data()
    {
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < observations[i]; ++j)
            {
                delete[] data[i][j];
            }
            delete[] data[i];
        }

        delete[] data;
    }

    void set(size_t x, size_t y, size_t z, float n){
        data[x][y][z]=n;
    }

    float get(size_t x, size_t y, size_t z){
        return data[x][y][z];
    }

    void print()
    {
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < observations[i]; j++)
            {
                std::cout << "(";
                for (size_t k = 0; k < v; k++)
                {
                    std::cout << data[i][j][k] << " ";
                }
                std::cout << ") ";
            }
            std::cout << std::endl;
        }
    }

    size_t getpeople(){
        return n;
    }

    //returns the number of observations for a certain person i
    size_t get_atoms(size_t i){
        return observations[i];
    }
};

arma::vec toArma(float*array, size_t arraySize){
    return armaVector(array, arraySize, false);
}

void toStd(arma::vec vector, float **array, size_t*size){
    *array=vector.memptr();
    *size=vector.size();
}

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


private:
    Data data; //input
    Dimensions dim; //input
    vec alpha; // (K x 1) K:= numero di DC (passato in input)
    vec pi; // (K x 1) pesi per scegliere DC
    vec beta; // (L x 1) L := numero di OC (passato in input)
    mat w; // (L x K) scelto K, pesi per scegliere OC
    arma::Col<int> S; // (J x 1) assegna un DC ad ogni persona (J persone)
    Mat<int> M; // (J x max(n_j)) assegna per ogni persona un OC ad ogni atomo di quella persona (J persone e n_j atomo per persona j)
    Theta theta; // (L x 1) ogni elemento di Theta è uno degli atomi comuni a tutti i DC


public:

    // Costruttore 
    Chain(const Dimensions& input_dim, const vec& input_alpha, const vec& input_beta, Data& input_data) : dim(input_dim), alpha(input_alpha), beta(input_beta), data(input_data) {

        // Generazione dei pesi pi e w
        pi = draw_pi(alpha); // genera i pesi per DC dalla Dirichlet
        w = draw_W(beta, dim.K);

        // Generazione delle variabili categoriche S e M
        S = draw_S(pi, dim.J);
        M = draw_M(w, S, dim.N, dim.J);

        // Generazione dei parametri theta
        theta = draw_theta();
    }


    void chain_step(void) {
        alpha = update_alpha(alpha, S, dim.K);
        beta = update_beta(beta, M, dim.L);

        pi = draw_pi(alpha);
        pi = update_pi(pi, w, dim.K, dim.L);

        w = draw_W(beta, dim.K);
        w = update_W();


    }


    // Stampa
    void print(void) {
        cout << "pi:\n" << pi << endl;
        cout << "w:\n" << w << endl;
        cout << "S:\n" << S << endl;
        cout << "M:\n" << M << endl;
    }
};


vec draw_pi(vec& alpha) {
    return generateDirichlet(alpha);
};

mat draw_W(vec& beta, int& K) {
    mat w;
    for (int k = 0; k < K; k++) {
            vec w_k = generateDirichlet(beta); //genera un vettore dalla dirichlet per ogni k
            w = arma::join_rows(w, w_k); // costruisce la matrice dei pesi aggiungendo ogni colonna
        }
    return w;
};

arma::Col<int> draw_S(vec& pi, int J) {
    arma::Col<int> S;
    S.set_size(J);
    std::default_random_engine generatore_random;
    discrete_distribution<int> Cat(pi.begin(), pi.end()); 
    for (int j = 0; j < J; j++) {
        S(j) = Cat(generatore_random);
    }
    return S;
};

arma::Mat<int> draw_M(mat& w, arma::Col<int>& S, int& N, int& J)  {
    arma::Mat<int> M;
    M.set_size(J, N);
    std::default_random_engine generatore_random;
    for (int j = 0; j < J; ++j) {
        discrete_distribution<int> Cat_w(w.col(S(j)).begin(), w.col(S(j)).end());
        for (int i = 0; i < N; i++) {
            M(j,i) = Cat_w(generatore_random); // genera M(matrice J x N) per ogni persona (j), genera N valori da categoriche seguendo la distribuzione Cat_w
        }
    }
    return M;
};


vec update_alpha(vec& alpha, arma::Col<int>& S, int& K) {
    for (int k = 0; k < K; ++k) {
        alpha(k) = alpha(k) + arma::accu(S == k);
    }
    return alpha;
};

vec update_beta(vec& beta, arma::Mat<int>& M, int& L) {
    for (int l = 0; l < L; ++l) {
        beta(l) = beta(l) + arma::accu(M == l);
    }
};

vec update_pi(vec& pi, mat& w, int& K, int& L) { // guarda se funziona mettendo il log, senno integra via M
    for (int k = 0; k < K; ++k) {
        int prod = 1;
        for (int l = 0; l < L; ++l) {
            prod *= w(l,k);
        }
        pi(k) = pi(k) * prod;
    }
    return pi;
};

mat update_w() {

};

// NON CI SONO DIRICHLET PROCESS
// usare cholesky per estrarre dalla wishart
