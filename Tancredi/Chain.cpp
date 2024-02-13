//
// Created by super on 05/02/2024.
//
#include <iostream>
#include "Chain.hpp"
/*
#include <RInside.h>
#include <Rcpp.h>
*/
#include <armadillo>

/* R(argc, argv)  da mettere nel costruttore*/

Chain::Chain(const Dimensions& input_dim, const vec& input_alpha, const mat& input_beta, Data& input_data, vec& mu, double& lambda, int& nu, mat& scale_matrix, int argc, char *argv[]): dim(input_dim), alpha(input_alpha), beta(input_beta), data(input_data), theta(dim.L, dim.v), log_W(input_dim.L, input_dim.K), S(input_dim.J), M(input_dim.max_N, input_dim.J){

    // Generazione dei pesi pi e w
    draw_log_pi(alpha); // genera i pesi per DC dalla Dirichlet
    draw_log_W(beta);
    // Generazione delle variabili categoriche S e M
    draw_S();

    M.fill(-1);
    draw_M();
    // Generazione dei parametri theta
    draw_theta(mu, lambda, nu, scale_matrix);
}

void Chain::chain_step(void) {
    //UPDATE DISTRIBUTIONAL CLUSTERS
        // UPDATE PI (prima aggiorno i pesi e poi i nuovi valori di pi)
        vec alpha_post = update_pi();
        draw_log_pi(alpha_post);
        update_S();

    // UPDATE OBSERVATIONAL CLUSTERS
        // UPDATE OMEGA (prima aggiorno i pesi poi i nuovi valori di omega)
        mat beta_post = update_omega();
        draw_log_W(beta_post);

        // UPDATE M (prima aggiorno i pesi e poi i nuovi valori di M)
        update_M();

    // UPDATE PARAMETERS
        update_theta_NIW(); // PROVA CON NIW PER VEDERE SE FUNZIONA

}

void Chain::print(void) {
    cout << "pi:\n" << arma::normalise(arma::exp(log_pi),1) << endl;
    cout << "w:\n" << arma::normalise(arma::exp(log_W),1, 0) << endl;
    cout << "M:\n" << M << endl;
    //cout << "alpha:\n" << alpha << endl;
    //cout << "Beta:\n" << beta << endl;
    cout << "S:\n" << S << endl;
    cout << "Theta:\n" << endl;
    theta.print();
}

void Chain::draw_log_pi(vec& a) {
    vec pi = generateDirichlet(a);
    log_pi =  arma::log(pi);
};

// CONTROLLARE CHE NON RIUTILIZZI BETA PRECEDENTE MA SOLO IL PRIMO STESSA COS


void Chain::draw_log_W(mat& B) {
    for (int k = 0; k < dim.K; k++) {
        vec log_w_k = arma::log(generateDirichlet(B.col(k))); //genera un vettore dalla dirichlet per ogni k
        log_W.col(k) = log_w_k; // costruisce la matrice dei pesi aggiungendo ogni colonna
    }
};


void Chain::draw_S(void) {
    arma::vec pi = arma::exp(log_pi);
    std::default_random_engine generatore_random;
    discrete_distribution<int> Cat(pi.begin(), pi.end());
    for (int j = 0; j < dim.J; j++) {
        S(j) = Cat(generatore_random);
    }
};

void Chain::draw_M(void)  {
    mat w = arma::exp(log_W);
    std::default_random_engine generatore_random;
    for (int j = 0; j < dim.J; ++j) {
        discrete_distribution<int> Cat_w(w.col(S(j)).begin(), w.col(S(j)).end());
        for (int i = 0; i < *(dim.N+j); i++) {
            M(i,j) = Cat_w(generatore_random); // genera M(matrice J x N) per ogni persona (j), genera N valori da categoriche seguendo la distribuzione Cat_w
        }
    }
};

void Chain::draw_theta(vec& mu, double& lambda, int& nu, mat& scale_matrix) {
    for (size_t l = 0; l < dim.L; ++l) {
        mat covariance = generateRandomMatrix(nu, scale_matrix);
        vec mean = generateRandomVector(mu, covariance * (1/lambda));
        theta.set_mean(l, mean);
        theta.set_covariance(l,covariance);
        theta.set_mu(l,mu);
        theta.set_lambda(l,lambda);
        theta.set_nu(l,nu);
        theta.set_scale_matrix(l, scale_matrix);
    }
};

vec Chain::update_pi(void) {
    vec alpha_post = alpha;
    for (size_t k = 0; k < dim.K; ++k) {
        alpha_post(k) += arma::accu(S == k);
    }
    return alpha_post;
};


mat Chain::update_omega(void) {
    mat beta_post = beta;
    for (size_t k = 0; k < dim.K; ++k) {
        arma::mat Mk = M.cols(arma::find(S == k));
        for (size_t l = 0; l < dim.L; ++l) {
            beta_post(l,k) += arma::accu(Mk == l);
        }
    }
    return beta_post;
};

void Chain::update_S(void) { // guarda se funziona mettendo il log, senno integra via M
    std::default_random_engine generatore_random;
    size_t* observations = data.get_observationsFor();
    for (size_t j = 0; j < dim.J; ++j){
        vec log_P(dim.K,arma::fill::zeros);
        for (size_t k = 0; k < dim.K; ++k) {
            float log_Pk = log_pi(k);
                for (size_t i = 0; i < *(observations + j); ++i) {
                    log_Pk += log_W(M(i,j),k);
                }
            log_P(k) = log_Pk;
        }
        vec pi = arma::exp(log_P- arma::max(log_P)); // - arma::max(log_P)
        pi = arma::normalise(pi, 1);
        discrete_distribution<int> Cat(pi.begin(), pi.end()); 
        S(j) = Cat(generatore_random);
    }
};


double logLikelihood(const vec& x, const vec& mean, const mat& covariance) {
    int dim = x.size();
    vec dif = x - mean;
    vec prod = arma::inv(covariance) * dif;
    double res = arma::dot(prod, dif);
    double normalization = log(abs(det(covariance)));
    double logLik = -(normalization + res);
    return logLik;
}


void Chain::update_M(void) {
    int J = dim.J;
    size_t* observations = data.get_observationsFor();
    for (size_t j = 0; j < J; ++j) {
        size_t k = S(j);
        for (size_t i = 0; i < *(observations + j); ++i) {
            vec log_Wk(dim.L, arma::fill::zeros);
            for (size_t l = 0; l < dim.L; ++l) {
                log_Wk(l) = log_W(l,k);
                log_Wk(l) += logLikelihood(data.get_vec(j,i), theta.get_mean(l), theta.get_cov(l));
            }
            vec Wk = arma::exp(log_Wk- arma::max(log_Wk)); // - arma::max(log_Wk)
            Wk = arma::normalise(Wk, 1);
            std::default_random_engine generatore_random;
            discrete_distribution<int> Cat(Wk.begin(), Wk.end()); 
            M(i,j) = Cat(generatore_random);
        }
    }
};


vec generateDirichlet(const vec& a) {

    int k = a.n_elem;
    vec gammaSample(k);

    // Genera k variabili casuali dalla distribuzione gamma
    for (int i = 0; i < k; ++i) {
        gammaSample(i) = randg(1, distr_param(a(i), 1.0))(0);
    }

    // Normalizza il campione per ottenere una variabile casuale Dirichlet
    return gammaSample / sum(gammaSample);
}

vec generateRandomVector(const vec& mean, const mat& covariance) {
    int dim = mean.size();
    vec randomVector = mean + chol(covariance, "lower") * arma::randn(dim);
    return randomVector;
}

mat generateRandomMatrix(int degreesOfFreedom, const mat& scaleMatrix) {
    mat randomMatrix = iwishrnd(scaleMatrix, degreesOfFreedom);
    return randomMatrix;
}
/*
void Chain::update_theta(void)
{
    vec** data_per_cluster = {nullptr};
    data_per_cluster = new vec*[dim.L];
    for (size_t l = 0; l < dim.L; l++) {
        uvec indeces = arma::find(M == l);
        data_per_cluster[l] = new vec[indeces.size()];
        for (size_t i = 0; i < indeces.size(); i++) {
            data_per_cluster[l][i] = data.get_vec(indeces(i) / dim.max_N, indeces(i) % dim.max_N);
        }
    }
    
    using namespace Rcpp;

 // Initialize RInside inside the function
    

    // Create an arma::cube to hold the arrays

    arma::cube arma_mu(theta.get_size_v(), 1, theta.size());

    // Gather arrays from the matrices into the cube
    for (size_t k = 0; k < theta.size(); ++k)
    {
        arma_mu.slice(k) = arma::reshape(theta.get_mean(k), theta.get_size_v(), 1); // reshape the vector into a matrix and assign it to the slice
    }

    // Create the cube
    arma::cube arma_sigma(theta.get_size_v(), theta.get_size_v(), theta.size());

    // Gather vectors into the cube
    for (size_t k = 0; k < theta.size(); ++k)
    {
        arma_sigma.slice(k) = theta.get_cov(k);
    }

    // Create the third cube
    arma::cube arma_A(theta.get_size_v(), theta.get_size_v(), theta.size());

     for (size_t k = 0; k < theta.size(); ++k)
    {
        arma_A.slice(k) = theta.get_DAG(k);
    }

    int nLayers_mu = arma_mu.n_slices, nLayers_sigma = arma_sigma.n_slices, nLayers_A = arma_A.n_slices;
    int nRows_mu = arma_mu.n_rows, nRows_sigma = arma_sigma.n_rows, nRows_A = arma_A.n_rows;
    int nCols_mu = arma_mu.n_cols, nCols_sigma = arma_sigma.n_cols, nCols_A = arma_A.n_cols;

    // Rcpp::Rcout << arma_mu << std::endl;
    // Rcpp::Rcout << arma_sigma << std::endl;
    // Rcpp::Rcout << arma_A << std::endl;

    // Create lists to represent the cubes
    List Rcpp_mu(nLayers_mu);
    List Rcpp_sigma(nLayers_sigma);
    List Rcpp_A(nLayers_A);

    // Populate the first cube
    for (int k = 0; k < nLayers_mu; ++k)
    {
        NumericMatrix layer1(nRows_mu, nCols_mu);

        for (int i = 0; i < nRows_mu; ++i)
        {
            for (int j = 0; j < nCols_mu; ++j)
            {
                layer1(i, j) = arma_mu(i, j, k);
            }
        }

        Rcpp_mu[k] = layer1;
    }

    // Populate the second cube
    for (int k = 0; k < nLayers_sigma; ++k)
    {
        NumericMatrix layer2(nRows_sigma, nCols_sigma);

        for (int i = 0; i < nRows_sigma; ++i)
        {
            for (int j = 0; j < nCols_sigma; ++j)
            {
                layer2(i, j) = arma_sigma(i, j, k);
            }
        }

        Rcpp_sigma[k] = layer2;
    }

    // Populate the third cube
    for (int k = 0; k < nLayers_A; ++k)
    {
        NumericMatrix layer3(nRows_A, nCols_A);

        for (int i = 0; i < nRows_A; ++i)
        {
            for (int j = 0; j < nCols_A; ++j)
            {
                layer3(i, j) = arma_A(i, j, k);
            }
        }

        Rcpp_A[k] = layer3;
    }

    // Pass all three cubes to the R function
    R["mu_mat"] = Rcpp_mu;
    R["sigma_mat"] = Rcpp_sigma;
    R["A_mat"] = Rcpp_A;

    vec obs_in_OC(dim.L, arma::fill::zeros);
    for (size_t l = 0; l < dim.L; ++l) {
        obs_in_OC(l) = arma::accu(M == l);
    }

    NumericVector n(obs_in_OC.n_elem);

    // Copy the elements from the arma::vec to the Rcpp::NumericVector
    for (size_t i = 0; i < obs_in_OC.n_elem; ++i) {
        n[i] = obs_in_OC(i);
    }

    Rcpp::Rcout << n << std::endl;


    // Define an integer
    int L = theta.size();
    int q = theta.get_size_v();
    // Pass the vector and integer to R
    R["n"] = n;
    R["L"] = L;
    R["q"] = q;

    R.parseEvalQ("source('mixture_dag.R')");
    R.parseEvalQ("result <- mixture_dags(mu_mat, sigma_mat, A_mat, L, q, n)"); // AGGIUNGERE Y E MATRICE M DEGLI OBS CLUSTERS (OPPURE I DATI GIA DIVISI PER CLUSTER)

    // Retrieve and print the result cube
    size_t rows[3], cols[3];
    rows[0] = nRows_mu;
    rows[1] = nRows_sigma;
    rows[2] = nRows_A;
    cols[0] = nCols_mu;
    cols[1] = nCols_sigma;
    cols[2] = nCols_A;

    List result_list=R["result"];

    for (int l = 0; l < result_list.size(); ++l)
    {
        List layer_list = result_list[l];

        for (int k = 0; k < layer_list.size(); ++k)
        {
            if (l == 0)
            {
                NumericVector layer = layer_list[k];

                for (int i = 0; i < rows[l]; ++i)
                {
                    theta.set_m(k,i, layer(i));
                }
            }
            else if(l==1)
            {
                NumericMatrix layer = layer_list[k];

                for (int i = 0; i < rows[l]; ++i)
                {
                    for (int j = 0; j < cols[l]; ++j)
                    {
                        theta.set_c(k, i, j, layer(i,j));
                    }
                }
            }
            else
            {
                NumericMatrix layer = layer_list[k];

                for (int i = 0; i < rows[l]; ++i)
                {
                    for (int j = 0; j < cols[l]; ++j)
                    {
                        theta.set_d(k, i, j, layer(i,j));
                    }
                }
            }
            
        }
    }

    // theta.print();

}
*/

void Chain::update_theta_NIW() {

    vec n(dim.L, arma::fill::zeros);

    for (size_t l = 0; l < dim.L; ++l) {
        n(l) = arma::accu(M == l);
    }

    for (size_t l = 0; l < dim.L; ++l) {
        if (n(l) != 0) {
            uvec indeces = arma::find(M == l);
            mat data_cluster_l(indeces.size(), theta.get_size_v());
            for (size_t i = 0; i < indeces.size(); i++) {
                data_cluster_l.row(i) = data.get_vec(indeces(i) / dim.max_N, indeces(i) % dim.max_N).t();
            }
            vec x_bar = arma::mean(data_cluster_l, 0).t();
            mat x_bar_mat = arma::repmat(x_bar, 1, n(l));

            // aggiorna i parametri della NIW
            double lambda_post = theta.get_lambda(l) + n(l);
            int nu_post = theta.get_nu(l) + n(l);

            // covariance post 
                // S
                mat S = (data_cluster_l - x_bar_mat.t()).t() *(data_cluster_l - x_bar_mat.t());
                // T
                mat T = ((x_bar - theta.get_mu(l)) * (x_bar - theta.get_mu(l)).t()) * (theta.get_lambda(l) * n(l)) / lambda_post;

            mat scale_matrix_post = theta.get_scale_matrix(l) + S + T;

            vec mu_post = (theta.get_lambda(l) * theta.get_mu(l) + n(l) * x_bar) / (lambda_post);
           

            // UPDATE NIW
            theta.set_covariance(l, generateRandomMatrix(nu_post, scale_matrix_post));
            theta.set_mean(l, generateRandomVector(mu_post, theta.get_cov(l) / lambda_post));
        }
        else {
            theta.set_covariance(l, generateRandomMatrix(theta.get_nu(l), theta.get_scale_matrix(l)));
            theta.set_mean(l, generateRandomVector(theta.get_mu(l), theta.get_cov(l) / theta.get_lambda(l)));

        }
    }
}



// package salso in R

/*
      // new parameters
      new_nu0 = nu0 + n_l ;
      new_beta0 = beta0+n_l ;
      new_m0 = (beta0 * m0 + n_l*ybar_l)/new_beta0;
      new_iW0 = iW0 + S + n_l*beta0/(new_beta0) * (ybar_l-m0).t() * (ybar_l-m0) ;
      out_W.slice(l) = rwish_cpp(arma::inv_sympd(new_iW0),new_nu0);
      tout_mu.col(l) = rmvtnorm_cpp_precision( new_m0.t(), new_beta0 * out_W.slice(l) ) ;

    } else {
      // from prior
      out_W.slice(l) = rwish_cpp(W0,nu0);
      tout_mu.col(l) = rmvtnorm_cpp_precision( m0.t(), beta0 * out_W.slice(l) )  ;
    }
  }
*/