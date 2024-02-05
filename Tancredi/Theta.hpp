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
    double*mean;
    double**covariance;
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

    void set(size_t s, double *mean, double **covariance);

    Parameters getParametersOf(size_t i);

    double *getMean(size_t i);

    double **getCovariate(size_t i);
};


#endif //A_BAYESIAN_PROJECT_THETA_HPP
