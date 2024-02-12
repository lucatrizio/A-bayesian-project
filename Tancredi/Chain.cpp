//
// Created by super on 05/02/2024.
//
#include <iostream>
#include "Chain.hpp"
#include <RInside.h>
#include <Rcpp.h>
#include <armadillo>



Chain::Chain(const Dimensions& input_dim, const vec& input_alpha, const mat& input_beta, Data& input_data, int argc, char *argv[]): dim(input_dim), alpha0(input_alpha), beta0(input_beta), data(input_data), theta(dim.L, dim.v), R(argc, argv) {

    // Generazione dei pesi pi e w
    log_pi = draw_log_pi(alpha0); // genera i pesi per DC dalla Dirichlet
    log_W = draw_log_W(beta0, dim.K);

    // Generazione delle variabili categoriche S e M
    S = draw_S(log_pi, dim.J);
    M = draw_M(log_W, S, dim.N, dim.J, dim.max_N);

    // Generazione dei parametri theta
    theta = draw_theta(dim.L, dim.v);
}

void Chain::chain_step(void) {
    //UPDATE DISTRIBUTIONAL CLUSTERS
        // UPDATE PI (prima aggiorno i pesi e poi i nuovi valori di pi)
        vec alpha = update_pi(alpha0, S, dim.K);
        log_pi = draw_log_pi(alpha);

        S = update_S(S, log_pi, log_W, dim.K, M, dim.J, data.get_observationsFor());


    // UPDATE OBSERVATIONAL CLUSTERS
        // UPDATE OMEGA (prima aggiorno i pesi poi i nuovi valori di omega)
        mat beta = update_omega(beta0, M, dim.L, dim.K, S);
        log_W  = draw_log_W(beta, dim.K);

        // UPDATE M (prima aggiorno i pesi e poi i nuovi valori di M)
        M = update_M(log_W, dim.L, dim.K, theta, data, S, M, dim.max_N, data.get_observationsFor());

    // UPDATE PARAMETERS
        // UPDATE THETA (da chiamare su R)
        theta = update_theta(theta); // PROVA CON NIW PER VEDERE SE FUNZIONA


}

void Chain::print(void) {
    cout << "pi:\n" << arma::normalise(arma::exp(log_pi),1) << endl;
    cout << "w:\n" << arma::normalise(arma::exp(log_W),1, 0) << endl;
    cout << "S:\n" << S << endl;
    cout << "M:\n" << M << endl;
    theta.print();
}

vec Chain::draw_log_pi(vec& alpha) {
    vec pi = generateDirichlet(alpha);
    return arma::log(pi);
};

// NORMALIZZARE? a posto in teoria
mat Chain::draw_log_W(mat& beta, size_t& K) {
    mat log_W;
    for (int k = 0; k < K; k++) {
        vec log_Wk = arma::log(generateDirichlet(beta.col(k))); //genera un vettore dalla dirichlet per ogni k
        log_W = arma::join_rows(log_W, log_Wk); // costruisce la matrice dei pesi aggiungendo ogni colonna
    }
    return log_W;
};


vec Chain::draw_S(vec& log_pi, size_t& J) {
    arma::vec pi = arma::exp(log_pi);
    vec S;
    S.set_size(J);
    std::default_random_engine generatore_random;
    discrete_distribution<int> Cat(pi.begin(), pi.end());
    for (int j = 0; j < J; j++) {
        S(j) = Cat(generatore_random);
    }
    return S;
};


mat Chain::draw_M(mat& log_W, vec& S, size_t* N, size_t& J, size_t& max_N)  {
    mat w = arma::exp(log_W);
    mat M;
    M.ones(max_N, J);
    M=M*(-1);
    std::default_random_engine generatore_random;
    for (int j = 0; j < J; ++j) {
        discrete_distribution<int> Cat_w(w.col(S(j)).begin(), w.col(S(j)).end());
        for (int i = 0; i < *(N+j); i++) {
            M(i,j) = Cat_w(generatore_random); // genera M(matrice J x N) per ogni persona (j), genera N valori da categoriche seguendo la distribuzione Cat_w
        }
    }
    return M;
};

Theta Chain::draw_theta(size_t& L, size_t& v) {
    // estrai da una NIW
    Theta theta(L,v);

    vec mean0(v, arma::fill::randu);  // Vettore delle medie con valori casuali
    mat cov0 = arma::eye<arma::mat>(v, v);  // Matrice di covarianza come matrice identità
    mat scale_mat = 0.1 * arma::eye<arma::mat>(v, v);  // Matrice di scala come 0.1 * matrice identità
    double df = 5.0;  // Gradi di libertà

    for (size_t l = 0; l < L; ++l) {
        vec mu = generateRandomVector(mean0, cov0); // DA DECIDERE I PARAMETRI DELLA PRIOR DELLA NIW
        mat cov = generateRandomMatrix(df, scale_mat);
        theta.set_mean(l, mu);
        theta.set_covariance(l,cov);
    }

    return theta;
};

vec Chain::update_pi(vec& alpha, vec& S, size_t& K) {
    for (int k = 0; k < K; ++k) {
        alpha(k) = alpha(k) + arma::accu(S == k);
    }
    return alpha;
};


mat Chain::update_omega(mat& beta, mat M, size_t& L, size_t& K, vec& S) {
    for (int k = 0; k < K; ++k) {
        for (int l = 0; l < L; ++l) {
            arma::mat Mk = M.cols(arma::find(S == k));
            beta(l,k) += arma::accu(Mk == l);
        }
    }
    return beta;
};


vec Chain::update_S(vec& S, vec& log_pi, mat& log_W, size_t& K, mat& M, size_t& J, size_t* observations) { // guarda se funziona mettendo il log, senno integra via M
    std::default_random_engine generatore_random;
    for (int j = 0; j < J; ++j){
        vec log_P(K,arma::fill::zeros);
        for (int k = 0; k < K; ++k) {
            float log_Pk = log_pi(k);
                for (int i = 0; i < *(observations + j); ++i) {
                    log_Pk += log_W(M(i,j),k);
                }
            log_P(k) = log_Pk;
        }
        vec pi = arma::exp(log_P);
        pi = arma::normalise(pi, 1);
        discrete_distribution<int> Cat(pi.begin(), pi.end()); 
        S(j) = Cat(generatore_random);
    }
    return S;
};


double logLikelihood(const vec& x, const vec& mean, const mat& covariance) {
    int dim = x.size();
    double expTerm = -0.5 * as_scalar(trans(x - mean) * arma::inv(covariance) * (x - mean));
    double normalization = -0.5 * dim * log(2.0 * M_PI) - 0.5 * log(det(covariance));
    double logLik = normalization + expTerm;
    return logLik;
}


mat Chain::update_M(mat& log_W, size_t& L, size_t& K, Theta& theta, Data& data, vec& S, mat& M, size_t& N, size_t* observations) {
    int J = data.getNumPeople();
    for (int j = 0; j < J; ++j) {
        int k = S(j);
        for (int i = 0; i < *(observations + j); ++i) {
            vec log_Wk(L, arma::fill::zeros);
            for (size_t l = 0; l < L; ++l) {
                log_Wk(l) = log_W(l,k);
            }
            for (int l = 0; l < L; ++l) {
                log_Wk(l) += logLikelihood(data.get_vec(j,i), theta.get_mean(l), theta.get_cov(l));
            }
            vec Wk = arma::exp(log_Wk);
            Wk = arma::normalise(Wk, 1);
            /*
            std::cout << "==========" << std::endl;
            std::cout << "Wk vector for:(" <<i<<", "<<j<< ") " <<std::endl;
            Wk.print();
            std::cout << "==========" << std::endl;
            */
            std::default_random_engine generatore_random;
            discrete_distribution<int> Cat(Wk.begin(), Wk.end()); 
            M(i,j) = Cat(generatore_random);
        }
    }
    return M;
};


vec generateDirichlet(const vec& alpha) {

    int k = alpha.n_elem;
    vec gammaSample(k);

    // Genera k variabili casuali dalla distribuzione gamma
    for (int i = 0; i < k; ++i) {
        gammaSample(i) = randg(1, distr_param(alpha(i), 1.0))(0);
    }

    // Normalizza il campione per ottenere una variabile casuale Dirichlet
    return gammaSample / sum(gammaSample);
}

vec generateRandomVector(const vec& mean, const mat& covariance) {
    int dim = mean.size();
    vec randomVector = mean + chol(covariance, "lower") * randn<vec>(dim);
    return randomVector;
}

mat generateRandomMatrix(int degreesOfFreedom, const mat& scaleMatrix) {
    mat randomMatrix = iwishrnd(scaleMatrix, degreesOfFreedom);
    return randomMatrix;
}

Theta Chain::update_theta(Theta &theta)
{
    vec** data_per_cluster = {nullptr};
    data_per_cluster = new vec*[dim.L];
    for (size_t l = 0; l < dim.L; l++) {
        std::cout << " ========= " << "M" << " ===========" << std::endl;
        M.print();
        uvec indeces = arma::find(M == l);
        std::cout << " ========= " << l << " ===========" << std::endl;
        indeces.print();
        data_per_cluster[l] = new vec[indeces.size()];
        for (size_t i = 0; i < indeces.size(); i++) {
            data_per_cluster[l][i] = data.get_vec(indeces(i) / dim.max_N, indeces(i) % dim.max_N);
        }
        for (size_t i = 0; i < indeces.size(); i++) {
            std::cout << data_per_cluster[l][i] << std::endl;
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

    return theta;
}