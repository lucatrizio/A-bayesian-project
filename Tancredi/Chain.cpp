//
// Created by super on 05/02/2024.
//

#include "Chain.hpp"

Chain::Chain(const Dimensions& input_dim, const vec& input_alpha, const vec& input_beta, Data& input_data) : dim(input_dim), alpha(input_alpha), beta(input_beta), data(input_data), theta(dim.L, dim.v) {

    // Generazione dei pesi pi e w
    log_pi = draw_log_pi(alpha); // genera i pesi per DC dalla Dirichlet
    log_W = draw_log_W(beta, dim.K);

    // Generazione delle variabili categoriche S e M
    S = draw_S(log_pi, dim.J);
    M = draw_M(log_W, S, dim.N, dim.J, dim.max_N);

    // Generazione dei parametri theta
    theta = draw_theta(dim.L, dim.v);

}

void Chain::chain_step(void) {
    //UPDATE DISTRIBUTIONAL CLUSTERS
        // UPDATE PI (prima aggiorno i pesi e poi i nuovi valori di pi)
        alpha = update_pi(alpha, S, dim.K);
        log_pi = draw_log_pi(alpha);

        S = update_S(log_pi, log_W, dim.K, M, dim.J, data.get_observationsFor());

    // UPDATE OBSERVATIONAL CLUSTERS
        // UPDATE OMEGA (prima aggiorno i pesi poi i nuovi valori di omega)
        beta = update_omega(beta, M, dim.L, dim.K, S);
        log_W  = draw_log_W(beta, dim.K);
        // UPDATE M (prima aggiorno i pesi e poi i nuovi valori di M)

        M = update_M(log_W, dim.L, dim.K, theta, data, S, M, dim.max_N, data.get_observationsFor());
    
    // UPDATE PARAMETERS
        // UPDATE THETA (da chiamare su R)
        theta = update_theta(theta, data);

}

void Chain::print(void) {
    cout << "pi:\n" << arma::exp(log_pi) << endl;
    cout << "w:\n" << arma::exp(log_W) << endl;
    cout << "S:\n" << S << endl;
    cout << "M:\n" << M << endl;
}

vec Chain::draw_log_pi(vec& alpha) {
    vec pi = generateDirichlet(alpha);
    return arma::log(pi);
};


mat Chain::draw_log_W(mat& beta, size_t& K) {
    mat log_W;
    for (int k = 0; k < K; k++) {
        vec log_w_k = arma::log(generateDirichlet(beta)); //genera un vettore dalla dirichlet per ogni k
        log_W = arma::join_rows(log_W, log_w_k); // costruisce la matrice dei pesi aggiungendo ogni colonna
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
    M.zeros(max_N, J);
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
            beta.col(k)(l) += arma::accu(M.cols(arma::find(S = k)) == l);
        }
    }
    return beta;
};


vec Chain::update_S(vec& log_pi, mat& log_W, size_t& K, mat& M, size_t& J, size_t* observations) { // guarda se funziona mettendo il log, senno integra via M
    vec S;
    S.set_size(J);
    for (int j = 0; j < J; ++j){
        vec log_P(K,0);
        for (int k = 0; k < K; ++k) {
            float log_Pk = 0;
                for (int i = 0; i < *(observations + j); ++i) {
                    log_Pk += log_W(M(i,j),k) + log_pi(k);
                }
            log_P(k) = log_Pk;
        }
        vec pi = arma::exp(log_P);
        std::default_random_engine generatore_random;
        discrete_distribution<int> Cat(pi.begin(), pi.end()); 
        S(j) = Cat(generatore_random);
    }
    return S;
};


double logLikelihood(const vec& x, const vec& mean, const mat& covariance) {
    int dim = x.size();
    double expTerm = -0.5 * as_scalar(trans(x - mean) * inv(covariance) * (x - mean));
    double normalization = -0.5 * dim * log(2.0 * M_PI) - 0.5 * log(det(covariance));
    double logLik = normalization + expTerm;
    return logLik;
}


mat Chain::update_M(mat& log_W, size_t& L, size_t& K, Theta& theta, Data& data, vec& S, mat& M, size_t& N, size_t* observations) {
    int J = data.getNumPeople();
    for (int j = 0; j < J; ++j) {
        int k = S(j);
        for (int i = 0; i < *(observations + j); ++i) {
            vec log_Wk;
            log_Wk.set_size(L);
            for (int l = 0; l < L; ++l) {
                log_Wk(l) = log_W(l,k) + logLikelihood(data.get_vec(i,j), theta.get_mean(l), theta.get_cov(l));
            }
            vec Wk = arma::exp(log_Wk);
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