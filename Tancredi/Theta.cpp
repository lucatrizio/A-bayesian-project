//
// Created by super on 05/02/2024.
//

#include "Theta.hpp"

Theta::Theta(size_t L, size_t v) : size_L(L), size_v(v)
{
    theta = new Parameters[L];

    for (size_t i = 0; i < L; i++)
    {
        theta[i].mean = new double[v];
        theta[i].covariance = new double *[v];

        for (size_t j = 0; j < v; j++)
        {
            theta[i].covariance[j] = new double[v];
        }

    }

}

Theta::~Theta()
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

void Theta::print(){
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

void Theta::set(size_t s, double *mean, double **covariance)
{

    for (size_t i = 0; i < size_v; i++)
    {
        theta[s].mean[i] = mean[i];

        for (size_t j = 0; j < size_v; j++)
        {
            theta[s].covariance[i][j] = covariance[i][j];
        }
    }
}

Parameters Theta::getParametersOf(size_t i)
{
    return theta[i];
}

double Theta::*getMean(size_t i)
{
    return theta[i].mean;
}

double Theta::**getCovariate(size_t i)
{
    return theta[i].covariance;
}