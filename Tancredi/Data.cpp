//
// Created by super on 05/02/2024.
//

#include "Data.hpp"

Data::Data(size_t n, size_t *observations, size_t v) : n(n), observations(observations), v(v)
{

    data = new double **[n];

    for (size_t i = 0; i < n; ++i)
    {
        data[i] = new double *[observations[i]];

        for (size_t j = 0; j < observations[i]; ++j)
        {
            data[i][j] = new double[v];
        }
    }
}

~Data::Data()
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

void Data::set(size_t x, size_t y, size_t z, double n){
    data[x][y][z]=n;
}

double* Data::get_vec(size_t x, size_t y){
    return data[x][y];
}

double Data::get(size_t x, size_t y, size_t z){
    return data[x][y][z];
}

void Data::print()
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

size_t Data::getpeople(){
    return n;
}

size_t Data::get_atom(size_t i){
    return observations[i];
}

size_t* Data::get_atoms() {
    return observations;
}