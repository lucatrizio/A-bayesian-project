#include <math.h>
#include "vector"
#include <armadillo>
#include <random>
#include <iostream>
#include <math.h>
using namespace std;
using namespace arma;

// TODO

// Sparse matrix for M? rows of different size
// Dimension struct -> N is not an integer but a vector

// update_M to be implemented (remeber to use log likelihood) 
// draw_theta to be implemented
// update_theta to be implemented (remember to use log likelihood)





class Theta{
private:
    struct Parameters
    {
        vec mean;
        mat covariance;
    };

    Parameters*theta; //vector of parameters(mean and coavariance)
    size_t size_L, size_v; //Number of L observational clusters (normal distributions each with a mean and a covariance)
    
public:
    // Constructor
    Theta(size_t L, size_t v) : size_L(L), size_v(v)
    {
        theta = new Parameters[L];
        /*
        for (size_t i = 0; i < L; i++)
        {
            theta[i].mean = new double[v];
            theta[i].covariance = new double *[v];

            for (size_t j = 0; j < v; j++)
            {
                theta[i].covariance[j] = new double[v];
            }
            
        }
        */
        for (size_t l = 0; l < L; ++l) {
            theta[l].mean.set_size(v);
            theta[l].covariance.set_size(v,v);
            
        }
        
    }

    // Destructor
    ~Theta() {
        
        delete[] theta;
    }

    void set_mean(size_t l, const vec& mu) {
        theta[l].mean = mu;
    }

    void set_covariance(int& l, const mat& cov) {
        theta[l].covariance = cov;
    }

    size_t size(){
        return size_L;
    }
    size_t size_v(){
        return size_v;
    }

    Parameters get(size_t i){
        return theta[i];
    }

    arma::vec get_mean(size_t i){
        return theta[i].mean;
    }

    arma::mat get_cov(size_t i){

        return theta[i].covariance;
    }
    std::vector<std::vector<int>> get_DAG(size_t i){
        return theta[i].DAG;
    }

};

class Data {
private:
    vec** data;
    size_t n, //number of people
    *observations, //number of observations for each person
    v; //number of observed parameters for each observation

public:
    // Constructor
    Data(size_t n, size_t *observations, size_t v) : n(n), observations(observations), v(v)
    {

        data = new vec *[n];

        for (size_t i = 0; i < observations[i]; ++i)
        {
            data[i] = new vec[observations[i]];
        }
    }

    // Destructor
    ~Data()
    {
        for (size_t i = 0; i < n; ++i)
        {
            delete[] data[i];
        }

        delete[] data;
    }

    void set_vec(size_t x, size_t y, vec& v){
        data[x][y]=v;
    }

    vec get_vec(size_t x, size_t y){
        return data[x][y];
    }

    void print()
    {
        for (size_t i = 0; i < n; ++i)
        {
            std::cout << "Row " << i << ":\n";
            for (size_t j = 0; j < observations[i]; ++j)
            {
                std::cout << data[i][j] << std::endl;
            }
        }
    }

    size_t getNumPeople(){
        return n;
    }

    //returns the number of observations for a certain person i
    size_t get_atom(size_t i){
        return observations[i];
    }

    size_t* get_observationsFor() {
        return observations;
    }

    size_t get_dimObservation() {
        return v;
    }
};

/*
//--------Technical comment--------
//memory is self managed, armadillo doesn't allocate, delete[] array in main after use!
arma::vec toArmaVec(double*array, size_t arraySize){
    return arma::vec(array, arraySize, false);
}

//--------Technical comment--------
//memory managed by armadillo, kept until out of scope
double* toStd(const arma::vec& vector, size_t*size){
    *size=vector.size();
    return vector.memptr();
    
}
*/




struct Dimensions{
    int J; //n of persons
    int K; //n of DC
    int L; //n of OC
    int V; //dimensions of one data
    int* N; //numero di osservazioni per persona
    int max_N;
}; //non  credo nj serva // si serve 


// funzione per generare campioni da una dirichlet di con parametro alpha (vettore)
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


// Genera casualmente un vettore da una distribuzione normale multivariata
vec generateRandomVector(const vec& mean, const mat& covariance) {
    int dim = mean.size();
    vec randomVector = mean + chol(covariance, "lower") * randn<vec>(dim);
    return randomVector;
}

// Genera casualmente una matrice da una distribuzione inverse wishart
mat generateRandomMatrix(int degreesOfFreedom, const mat& scaleMatrix) {
    mat randomMatrix = iwishrnd(scaleMatrix, degreesOfFreedom);
    return randomMatrix;
}




class Chain{


private:
    Data data; //input
    Dimensions dim; //input
    vec alpha; // (K x 1) K:= numero di DC (passato in input)
    vec log_pi; // (K x 1) pesi per scegliere DC
    mat beta; // (L x K) L := numero di OC (passato in input) x numero di DC
    mat log_W; // (L x K) scelto K, pesi per scegliere OC
    vec S; // (J x 1) assegna un DC ad ogni persona (J persone)
    mat M; // (max(n_j) x J) assegna per ogni persona un OC ad ogni atomo di quella persona (J persone e n_j atomo per persona j)
    Theta theta; // (L x 1) ogni elemento di Theta è uno degli atomi comuni a tutti i DC


public:

    // Costruttore 
    Chain(const Dimensions& input_dim, const vec& input_alpha, const vec& input_beta, Data& input_data) : dim(input_dim), alpha(input_alpha), beta(input_beta), data(input_data) {

        // Generazione dei pesi pi e w
        log_pi = draw_log_pi(alpha); // genera i pesi per DC dalla Dirichlet
        log_W = draw_log_W(beta, dim.K);

        // Generazione delle variabili categoriche S e M
        S = draw_S(log_pi, dim.J);
        M = draw_M(log_W, S, dim.N, dim.J, dim.max_N);

        // Generazione dei parametri theta
        theta = draw_theta(theta, dim.L);
    }


    void chain_step(void) {
    //UPDATE DISTRIBUTIONAL CLUSTERS
        // UPDATE PI (prima aggiorno i pesi e poi i nuovi valori di pi)
        alpha = update_pi(alpha, S, dim.K);
        log_pi = draw_log_pi(alpha);
        // UPDATE S (prima aggiorno i pesi e poi i nuovi valori di S)
        log_pi = update_S(log_pi, log_W, dim.K, M, dim.J, data.get_observationsFor());
        S = draw_S(log_pi, dim.J);

    // UPDATE OBSERVATIONAL CLUSTERS
        // UPDATE OMEGA (prima aggiorno i pesi poi i nuovi valori di omega)
        beta = update_omega(beta, M, dim.L, dim.K, S);
        log_W  = draw_log_W(beta, dim.K);
        // UPDATE M (prima aggiorno i pesi e poi i nuovi valori di M)
        log_W = update_M(log_W, dim.L, dim.K, theta, data, S, M);
        M = draw_M(log_W, S, dim.N, dim.J, dim.max_N);
    
    // UPDATE PARAMETERS
        // UPDATE THETA (da chiamare su R)
        theta = update_theta(theta, data);
    }


    // Stampa
    void print(void) {
        cout << "pi:\n" << arma::exp(log_pi) << endl;
        cout << "w:\n" << arma::exp(log_W) << endl;
        cout << "S:\n" << S << endl;
        cout << "M:\n" << M << endl;
    }
};


vec draw_log_pi(vec& alpha) {
    vec pi = generateDirichlet(alpha);
    return arma::log(pi);
};


mat draw_log_W(mat& beta, int& K) {
    mat log_W;
    for (int k = 0; k < K; k++) {
            vec log_w_k = arma::log(generateDirichlet(beta.col(k))); //genera un vettore dalla dirichlet per ogni k
            log_W = arma::join_rows(log_W, log_w_k); // costruisce la matrice dei pesi aggiungendo ogni colonna
        }
    return log_W;
};


vec draw_S(arma::vec& log_pi, int J) {
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


mat draw_M(mat& log_W, vec& S, int* N, int& J, int& max_N)  {
    mat w = arma::exp(log_W);
    arma::mat M;
    M.zeros(max_N,J);
    std::default_random_engine generatore_random;
    for (int j = 0; j < J; ++j) {
        discrete_distribution<int> Cat_w(w.col(S(j)).begin(), w.col(S(j)).end());
        for (int i = 0; i < *(N+j); i++) {
            M(i,j) = Cat_w(generatore_random); // genera M(matrice J x N) per ogni persona (j), genera N valori da categoriche seguendo la distribuzione Cat_w
        }
    }
    return M;
};


vec update_pi(vec& alpha, vec& S, int& K) {
    for (int k = 0; k < K; ++k) {
        alpha(k) = alpha(k) + arma::accu(S == k);
    }
    return alpha;
};


vec update_omega(mat& beta, mat M, int& L, int& K, vec& S) {
    for (int k = 0; k < K; ++k) {
        for (int l = 0; l < L; ++l) {
            beta.col(k)(l) += arma::accu(M.cols(arma::find(S = k)) == l);
        }
    }
};


vec update_S(vec& log_pi, mat& log_W, int& K, mat& M,int& J, size_t* observations) { // guarda se funziona mettendo il log, senno integra via M
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


double logLikelihood(const vec& x, const vec& mean, const mat& covariance) {
    int dim = x.size();
    double expTerm = -0.5 * as_scalar(trans(x - mean) * inv(covariance) * (x - mean));
    double normalization = -0.5 * dim * log(2.0 * M_PI) - 0.5 * log(det(covariance));
    double logLik = normalization + expTerm;
    return logLik;
}


mat update_M(mat& log_W, int& L, int& K, Theta& theta, Data& data, vec& S, mat& M) {
    int N = data.getNumPeople();
    mat log_WP((L,K),0);
    for (int l = 0; l < L; ++l) {
        int log_Wl = 0;
        for (int j = 0; j < data.getNumPeople(); ++j) {
            int log_Wli = 0;
            for (int i = 0; i < data.get_atom(j); ++i) {
                log_Wli += logLikelihood(data.get_vec(i,j), theta.get_mean(M(i,j)), theta.get_cov(M(i,j)) + log_W(M(i,j), S(j)));
            }
            log_Wl += log_Wli;
        }
        log_WP(l) = log_Wl;
    }
    return log_WP; 
};


Theta draw_theta(Theta& theta, int& L) {
    // estrai da una NIW
    for (int l = 0; l < L; ++l) {
        vec mu = generateRandomVector(mean0, cov0); // DA DECIDERE I PARAMETRI DELLA PRIOR DELLA NIW
        mat cov = generateRandomMatrix(df, scale_mat);
        theta.set_mean(l, mu);
        theta.set_covariance(l,cov);
    }
    return theta;
};


Theta update_Theta() {
    // aggiorna una NIW 
};

// NON CI SONO DIRICHLET PROCESS
// usare cholesky per estrarre dalla wishart

    
    /*
    ~Theta()
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
    
    void print(){
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
    */