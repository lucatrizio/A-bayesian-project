//
// Created by super on 05/02/2024.
//

#include "Theta.hpp"

Theta::Theta(size_t L, size_t v) : size_L(L), size_v(v){
    theta = new Parameters[L];

    for (size_t l = 0; l < L; ++l) {
        theta[l].mean.set_size(v);
        theta[l].covariance.set_size(v,v); 
        theta[l].DAG=arma::zeros<arma::mat>(v, v);
    }    
}

Theta::~Theta(){
    //delete[] theta;
}

void Theta::print(){
    for (int l = 0; l < size_L; ++l) {
        std::cout << "mean" << l << std::endl;
        theta[l].mean.print();
        std::cout << "covariance" << l <<  std::endl;
        theta[l].covariance.print();
        //std::cout << "DAG" << l <<  std::endl;
        //theta[l].DAG.print();
    }
}

void Theta::set_mean(size_t l, const vec& mu) {
    theta[l].mean = mu;
}

void Theta::set_mu(size_t l, const vec& mu) {
    theta[l].mu = mu;
}

void Theta::set_lambda(size_t l, const double& lambda) {
    theta[l].lambda = lambda;
}

void Theta::set_nu(size_t l, const int& nu) {
    theta[l].nu = nu;
}

void Theta::set_scale_matrix(size_t l, const mat& scale_matrix) {
    theta[l].scale_matrix = scale_matrix;
}

void Theta::set_m(size_t l, size_t s, double n) {
    theta[l].mean[s] = n;
}

void Theta::set_covariance(size_t l, const mat& cov) {
    theta[l].covariance = cov;
}

void Theta::set_c(size_t l, size_t s, size_t r, double n) {
    theta[l].covariance(s,r) = n;
}

void Theta::set_d(size_t l, size_t s, size_t r, double n) {
    theta[l].DAG(s,r) = n;
}

size_t Theta::size(){
    return size_L;
}

size_t Theta::get_size_v(){
    return size_v;
}

Parameters Theta::get(size_t i){
    return theta[i];
}

arma::vec Theta::get_mean(size_t i){
    return theta[i].mean;
}

arma::vec Theta::get_mu(size_t i){
    return theta[i].mu;
}

double Theta::get_lambda(size_t i){
    return theta[i].lambda;
}

arma::mat Theta::get_cov(size_t i){
    return theta[i].covariance;
}

int Theta::get_nu(size_t i){
    return theta[i].nu;
}

arma::mat Theta::get_scale_matrix(size_t i){
    return theta[i].scale_matrix;
}

arma::mat Theta::get_DAG(size_t i){
    return theta[i].DAG;
}