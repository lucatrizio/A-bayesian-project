//
// Created by super on 05/02/2024.
//

#ifndef A_BAYESIAN_PROJECT_CHAIN_HPP
#define A_BAYESIAN_PROJECT_CHAIN_HPP
#include "Theta.hpp"
#include "Data.hpp"


struct Dimensions{
    size_t J; //n of persons
    size_t K; //n of DC
    size_t L; //n of OC
    size_t v; //dimensions of one data
    size_t* N; //numero di osservazioni per persona
    size_t max_N;
}; //non  credo nj serva // si serve 


class Chain{

private:
    Data data; //input
    Dimensions dim; //input
    vec alpha; // (K x 1) K:= numero di DC (passato in input)
    vec log_pi; // (K x 1) pesi per scegliere DC
    mat beta; // (L x K) L := numero di OC (passato in input) x numero di DC
    mat log_W; // (L x K) scelto K, pesi per scegliere OC
    vec S; // (J x 1) assegna un DC ad ogni persona (J persone)
    mat M; // (max(n_j) x J) assegna per ogni persona un OC ad ogni atomo di quella persona (J persone e n_j atomo per persona j)
    Theta theta; // (L x 1) ogni elemento di Theta è uno degli atomi comuni a tutti i DC


public:

    // Costruttore
    Chain(const Dimensions& input_dim, const vec& input_alpha, const mat& input_beta, Data& input_data);

    // Chain step
    void chain_step(void);

    // Stampa
    void print(void);

    //Draws
    vec draw_log_pi(vec& alpha);
    mat draw_log_W(mat& beta, size_t& K);
    vec draw_S(vec& log_pi, size_t& J);
    mat draw_M(mat& log_W, vec& S, size_t* N, size_t& J, size_t& max_N);
    Theta draw_theta(size_t& L, size_t& v);

    //Updates
    vec update_pi(vec& alpha, vec& S, size_t& K);
    mat update_omega(mat& beta, mat M, size_t& L, size_t& K, vec& S);
    vec update_S(vec& S, vec& log_pi, mat& log_W, size_t& K, mat& M, size_t& J, size_t* observations);
    mat update_M(mat& log_W, size_t& L, size_t& K, Theta& theta, Data& data, vec& S, mat& M, size_t& N, size_t* observations);
    Theta update_theta(); //da chiamare con R

};

// funzione per valutare la likelihood in un determinato punto multivariato
double logLikelihood(const vec& x, const vec& mean, const mat& covariance);

// funzione per generare campioni da una dirichlet di con parametro alpha (vettore)
vec generateDirichlet(const vec& alpha);

// Genera casualmente un vettore da una distribuzione normale multivariata
vec generateRandomVector(const vec& mean, const mat& covariance);

// Genera casualmente una matrice da una distribuzione inverse wishart
mat generateRandomMatrix(int degreesOfFreedom, const mat& scaleMatrix);

/*
arma::vec toArma(double*array, size_t arraySize){
    return arma::vec(array, arraySize, false);
}
*/

/*
//--------Technical comment--------
//memory managed by armadillo, kept until out of scope
double* toStd(const arma::vec& vector, size_t*size){
    *size=vector.size();
    return vector.memptr();

}
*/

#endif //A_BAYESIAN_PROJECT_CHAIN_HPP
