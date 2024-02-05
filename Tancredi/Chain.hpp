//
// Created by super on 05/02/2024.
//

#ifndef A_BAYESIAN_PROJECT_CHAIN_HPP
#define A_BAYESIAN_PROJECT_CHAIN_HPP
#include "Theta.hpp"
#include "Data.hpp"


struct Dimensions{
    int J; //n of persons
    int K; //n of DC
    int L; //n of OC
    int V; //dimensions of one data
    int N; //numero di osservazioni per persona
}; //non  credo nj serva // si serve


class Chain{

private:
    Data data; //input
    Dimensions dim; //input
    vec alpha; // (K x 1) K:= numero di DC (passato in input)
    vec log_pi; // (K x 1) pesi per scegliere DC
    vec beta; // (L x 1) L := numero di OC (passato in input)
    mat log_W; // (L x K) scelto K, pesi per scegliere OC
    vec S; // (J x 1) assegna un DC ad ogni persona (J persone)
    mat M; // (J x max(n_j)) assegna per ogni persona un OC ad ogni atomo di quella persona (J persone e n_j atomo per persona j)
    Theta theta; // (L x 1) ogni elemento di Theta è uno degli atomi comuni a tutti i DC


public:

    // Costruttore
    Chain(const Dimensions& input_dim, const vec& input_alpha, const vec& input_beta, Data& input_data) : dim(input_dim), alpha(input_alpha), beta(input_beta), data(input_data);

    // Chain step
    void chain_step(void);


    // Stampa
    void print(void)

    //Draws
    vec draw_log_pi(vec& alpha);
    mat draw_log_W(vec& beta, int& K);
    vec draw_S(arma::vec& log_pi, int J);
    mat draw_M(mat& log_W, vec& S, int& N, int& J);
    Theta draw_theta(Theta& theta, int& L);

    //Updates
    vec update_pi(vec& alpha, vec& S, int& K);
    vec update_omega(vec& beta, mat M, int& L);
    vec update_S(vec& log_pi, mat& log_W, int& K, mat& M,int& J, size_t* observations);
    mat update_M(mat& log_W, int& L, int& K, Theta& theta, Data& data, vec& S, mat& M);
    Theta update_Theta(); //da fare

    double logLikelihood(const vec& x, const vec& mean, const mat& covariance);
};


// funzione per generare campioni da una dirichlet di con parametro alpha (vettore)
vec generateDirichlet(const vec& alpha);

// Genera casualmente un vettore da una distribuzione normale multivariata
vec generateRandomVector(const vec& mean, const mat& covariance);

// Genera casualmente una matrice da una distribuzione inverse wishart
mat generateRandomMatrix(int degreesOfFreedom, const mat& scaleMatrix);


arma::vec toArma(double*array, size_t arraySize){
    return arma::vec(array, arraySize, false);
}

//--------Technical comment--------
//memory managed by armadillo, kept until out of scope
double* toStd(const arma::vec& vector, size_t*size){
    *size=vector.size();
    return vector.memptr();

}

#endif //A_BAYESIAN_PROJECT_CHAIN_HPP
