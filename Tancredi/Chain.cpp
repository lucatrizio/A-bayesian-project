//
// Created by super on 05/02/2024.
//

#include "Chain.hpp"

Chain::Chain(const Dimensions& input_dim, const vec& input_alpha, const vec& input_beta, Data& input_data) : dim(input_dim), alpha(input_alpha), beta(input_beta), data(input_data) {

    // Generazione dei pesi pi e w
    log_pi = draw_log_pi(alpha); // genera i pesi per DC dalla Dirichlet
    log_W = draw_log_W(beta, dim.K);

    // Generazione delle variabili categoriche S e M
    S = draw_S(log_pi, dim.J);
    M = draw_M(log_W, S, dim.N, dim.J);

    // Generazione dei parametri theta
    draw_theta(theta, dim.L);
}

void Chain::chain_step(void) {
    alpha = update_pi(alpha, S, dim.K);
    beta = update_omega(beta, M, dim.L);

    log_pi = draw_log_pi(alpha);
    log_pi = update_S(log_pi, log_W, dim.K, M, dim.J, data.get_atoms());

    log_W  = draw_log_W(beta, dim.K);
    log_W = update_M(log_W, dim.L, dim.K, theta, data, S, M);

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


mat Chain::draw_log_W(vec& beta, int& K) {
    mat log_W;
    for (int k = 0; k < K; k++) {
        vec log_w_k = arma::log(generateDirichlet(beta)); //genera un vettore dalla dirichlet per ogni k
        log_W = arma::join_rows(log_W, log_w_k); // costruisce la matrice dei pesi aggiungendo ogni colonna
    }
    return log_W;
};


vec Chain::draw_S(arma::vec& log_pi, int J) {
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


mat Chain::draw_M(mat& log_W, vec& S, int& N, int& J)  {
    mat w = arma::exp(log_W);
    arma::mat M;
    M.zeros(J, N);
    std::default_random_engine generatore_random;
    for (int j = 0; j < J; ++j) {
        discrete_distribution<int> Cat_w(w.col(S(j)).begin(), w.col(S(j)).end());
        for (int i = 0; i < N; i++) {
            M(j,i) = Cat_w(generatore_random); // genera M(matrice J x N) per ogni persona (j), genera N valori da categoriche seguendo la distribuzione Cat_w
        }
    }
    return M;
};

Theta Theta::draw_theta(Theta& theta, int& L) {
    // estrai da una NIW
    for (int l = 0; l < L; ++l) {
        vec mu = generateRandomVector(mean0, cov0); // DA DECIDERE I PARAMETRI DELLA PRIOR DELLA NIW
        mat cov = generateRandomMatrix(df, scale_mat);
        theta.set_mean(l, mu);
        theta.set_covariance(l,cov);
    }
    return theta;
};

vec Chain::update_pi(vec& alpha, vec& S, int& K) {
    for (int k = 0; k < K; ++k) {
        alpha(k) = alpha(k) + arma::accu(S == k);
    }
    return alpha;
};


vec Chain::update_omega(vec& beta, mat M, int& L) {
    for (int l = 0; l < L; ++l) {
        beta(l) = beta(l) + arma::accu(M == l);
    }
};


vec Chain::update_S(vec& log_pi, mat& log_W, int& K, mat& M,int& J, size_t* observations) { // guarda se funziona mettendo il log, senno integra via M
    vec log_P(K,0);
    for (int k = 0; k < K; ++k) {
        float log_Pk = 0;
        for (int j = 0; j < J; ++j) {
            float log_lik = 0;
            for (int i = 0; i < *(observations + j); ++i) {
                log_lik += log_W(M(i,j),k) + log_pi(k);
            }
            log_Pk += log_lik;
        }
        log_P(k) = log_Pk;
    }
    return log_P;
};


double Chain::logLikelihood(const vec& x, const vec& mean, const mat& covariance) {
    int dim = x.size();
    double expTerm = -0.5 * as_scalar(trans(x - mean) * inv(covariance) * (x - mean));
    double normalization = -0.5 * dim * log(2.0 * M_PI) - 0.5 * log(det(covariance));
    double logLik = normalization + expTerm;
    return logLik;
}


mat Chain::update_M(mat& log_W, int& L, int& K, Theta& theta, Data& data, vec& S, mat& M) {
    int N = data.getpeople();
    mat log_WP((L,K),0);
    for (int l = 0; l < L; ++l) {
        int log_Wl = 0;
        for (int j = 0; j < data.getpeople(); ++j) {
            int log_Wli = 0;
            for (int i = 0; i < *(data.get_atoms() + j); ++i) {
                log_Wli += logLikelihood(data.get_vec(i,j), theta.getParametersOf(M(i,j)).mean, theta.getParametersOf(M(i,j)).covariance) + log_W(M(i,j), S(j));
            }
            log_Wl += log_Wli;
        }
        log_WP(l) = log_Wl;
    }
    return log_WP;
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