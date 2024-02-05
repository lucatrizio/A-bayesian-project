//
// Created by super on 05/02/2024.
//

#ifndef A_BAYESIAN_PROJECT_DATA_HPP
#define A_BAYESIAN_PROJECT_DATA_HPP
#include <math.h>
#include "vector"
#include <armadillo>
#include <random>
#include <iostream>
#include <math.h>
using namespace std;
using namespace arma;


class Data {
private:
    double***data;
    size_t n, //number of people
    *observations, //number of observations for each person
    v; //number of observed parameters for each observation

public:
    // Constructor
    Data(size_t n, size_t *observations, size_t v);

    // Destructor
    ~Data();

    void set(size_t x, size_t y, size_t z, double n);

    double get(size_t x, size_t y, size_t z);

    double* get_vec(size_t x, size_t y)

    void print();

    size_t getpeople();

    //returns the number of observations for a certain person i
    size_t get_atom(size_t i);

    size_t* get_atoms();
};



#endif //A_BAYESIAN_PROJECT_DATA_HPP
