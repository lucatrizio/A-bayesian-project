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
    vec mean;
    mat covariance;
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

    void set_mean(size_t& l, const vec& mu);

    void set_covariance(size_t& l, const mat& cov);

    size_t size();

    size_t get_size_v();

    Parameters get(size_t i);

    vec get_mean(size_t i);

    mat get_cov(size_t i);
};


#endif //A_BAYESIAN_PROJECT_THETA_HPP
