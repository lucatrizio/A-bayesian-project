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
    vec**data;
    size_t J, //number of people
    *observations, //number of observations for each person
    v; //number of observed parameters for each observation

public:
    // Constructor
    Data(size_t J_input, size_t* observations_input, size_t v_input);

    // Destructor
    ~Data();

    void set_vec(size_t x, size_t y, vec& v);

    vec get_vec(size_t x, size_t y);

    void print();

    size_t getNumPeople();

    //returns the number of observations for a certain person i
    size_t get_atom(size_t i);

    size_t* get_observationsFor();

    size_t get_dimObservation();
};



#endif //A_BAYESIAN_PROJECT_DATA_HPP
