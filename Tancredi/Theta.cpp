//
// Created by super on 05/02/2024.
//

#include "Theta.hpp"

Theta::Theta(size_t L, size_t v) : size_L(L), size_v(v){
    theta = new Parameters[L];

    for (size_t l = 0; l < L; ++l) {
        theta[l].mean.set_size(v);
        theta[l].covariance.set_size(v,v);      
    }    
}

Theta::~Theta(){
    delete[] theta;
}

void Theta::print(){

}

void Theta::set_mean(size_t& l, const vec& mu) {
    theta[l].mean = mu;
}

void Theta::set_covariance(size_t& l, const mat& cov) {
    theta[l].covariance = cov;
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

arma::mat Theta::get_cov(size_t i){
    return theta[i].covariance;
}

