//
// Created by super on 05/02/2024.
//

#include "Data.hpp"

Data::Data(size_t J_input, size_t* observations_input, size_t v_input) {
        J = J_input;
        observations = observations_input;
        v = v_input;
        data = new vec *[J];

        for (size_t i = 0; i < J; ++i)
        {
            data[i] = new vec[observations[i]];
        }
    }


Data::~Data()
    {
        
    }

void Data::set_vec(size_t x, size_t y, vec& v){
    data[x][y]=v;
}

vec Data::get_vec(size_t x, size_t y){
    return data[x][y];
}

void Data::print()
    {
        
        for (size_t i = 0; i < J; ++i)
        {
            std::cout << "Row " << i << ":\n";
            for (size_t j = 0; j < observations[i]; ++j)
            {
                std::cout << data[i][j] << std::endl;
            }
        }
    }

size_t Data::getNumPeople(){
    return J;
}

size_t Data::get_atom(size_t i){
    return observations[i];
}

size_t* Data::get_observationsFor() {
    return observations;
}

size_t Data::get_dimObservation() {
    return v;
}