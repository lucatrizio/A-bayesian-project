//
// Created by super on 05/02/2024.
//


#ifndef A_BAYESIAN_PROJECT_THETA_HPP
#define A_BAYESIAN_PROJECT_THETA_HPP
#include <math.h>
#include "vector"
#include <armadillo>
#include <random>
#include <iostream>
#include <math.h>
using namespace std;
using namespace arma;

struct Parameters
{
    // mean
    vec mean;
    vec mu;
    double lambda;

    // covariance matrix
    mat covariance;
    int nu;
    mat scale_matrix; // S^-1

    // dag
    mat DAG;
};

class Theta{
private:
    Parameters*theta; //vector of parameters(mean and coavariance)
    size_t size_L, size_v; //Number of L observational clusters (normal distributions each with a mean and a covariance)

public:
    // Constructor
    Theta(size_t L, size_t v);

    // Destructor
    ~Theta();

    void print();

    void set_mean(size_t l, const vec& mu);
    void set_mu(size_t l, const vec& mu);
    void set_lambda(size_t l, const double& lambda);

    void set_m(size_t l, size_t s, double n);
    void set_covariance(size_t l, const mat& cov);
    void set_nu(size_t l, const int& nu);
    void set_scale_matrix(size_t l, const mat& scale_matrix);
    
    void set_c(size_t l, size_t s, size_t r, double n);
    void set_d(size_t l, size_t s, size_t r, double n);

    size_t size();

    size_t get_size_v();

    Parameters get(size_t i);

    vec get_mean(size_t i);
    vec get_mu(size_t i);
    double get_lambda(size_t i);

    mat get_cov(size_t i);
    int get_nu(size_t i);
    mat get_scale_matrix(size_t i);

    mat get_DAG(size_t i);
};


#endif //A_BAYESIAN_PROJECT_THETA_HPP
