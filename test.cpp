#include <iostream>
#include <fstream>
#include <RInside.h>
#include <Rcpp.h>
#include <armadillo>

using namespace Rcpp;

void print(List RcppCube, size_t nLayer, size_t nCols, size_t nRows)
{

    for (size_t i = 0; i < nLayer; i++)
    {
        NumericMatrix mat = RcppCube[i];
        for (int j = 0; j < nRows; ++j)
        {
            for (int k = 0; k < nCols; ++k)
            {
                Rcpp::Rcout << mat(j, k) << " "; // Output each element
            }
            Rcpp::Rcout << std::endl; // Move to the next line after printing a row
        }
        Rcpp::Rcout << std::endl;
    }
}

void printR(List result_list, size_t *rows, size_t *cols)
{

    std::cout << "Result List:" << std::endl;

    for (int l = 0; l < result_list.size(); ++l)
    {
        List layer_list = result_list[l];
        std::cout << "Layer List " << l + 1 << ":" << std::endl;

        for (int k = 0; k < layer_list.size(); ++k)
        {
            if (l == 0)
            {
                NumericVector layer = layer_list[k];
                std::cout << "Layer " << k + 1 << ":" << std::endl;

                for (int i = 0; i < rows[l]; ++i)
                {
                    std::cout << layer(i) << std::endl;
                }
                std::cout << std::endl;
            }
            else
            {
                NumericMatrix layer = layer_list[k];
                std::cout << "Layer " << k + 1 << ":" << std::endl;

                for (int i = 0; i < rows[l]; ++i)
                {
                    for (int j = 0; j < cols[l]; ++j)
                    {
                        std::cout << layer(i, j) << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    RInside R(argc, argv);

    
        // Create the first cube
        arma::cube arma_mu(2, 1, 3, arma::fill::randn);

        // Create the second cube
        arma::cube arma_sigma(2, 2, 3, arma::fill::randn);

        // Create the third cube
        arma::cube arma_A(2, 2, 3, arma::fill::zeros);

        int nLayers_mu = arma_mu.n_slices, nLayers_sigma = arma_sigma.n_slices, nLayers_A = arma_A.n_slices;
        int nRows_mu = arma_mu.n_rows, nRows_sigma = arma_sigma.n_rows, nRows_A = arma_A.n_rows;
        int nCols_mu = arma_mu.n_cols, nCols_sigma = arma_sigma.n_cols, nCols_A = arma_A.n_cols;

        Rcpp::Rcout << arma_mu << std::endl;
        Rcpp::Rcout << arma_sigma << std::endl;
        Rcpp::Rcout << arma_A << std::endl;

        // Create lists to represent the cubes
        List Rcpp_mu(nLayers_mu);
        List Rcpp_sigma(nLayers_sigma);
        List Rcpp_A(nLayers_A);

        // Populate the first cube
        for (int k = 0; k < nLayers_mu; ++k) {
            NumericMatrix layer1(nRows_mu, nCols_mu);

            for (int i = 0; i < nRows_mu; ++i) {
                for (int j = 0; j < nCols_mu; ++j) {
                    layer1(i, j) = arma_mu(i, j, k);
                }
            }

            Rcpp_mu[k] = layer1;
        }

        // Populate the second cube
        for (int k = 0; k < nLayers_sigma; ++k) {
            NumericMatrix layer2(nRows_sigma, nCols_sigma);

            for (int i = 0; i < nRows_sigma; ++i) {
                for (int j = 0; j < nCols_sigma; ++j) {
                    layer2(i, j) = arma_sigma(i, j, k);
                }
            }

            Rcpp_sigma[k] = layer2;
        }

        // Populate the third cube
        for (int k = 0; k < nLayers_A; ++k) {
            NumericMatrix layer3(nRows_A, nCols_A);

            for (int i = 0; i < nRows_A; ++i) {
                for (int j = 0; j < nCols_A; ++j) {
                    layer3(i, j) = arma_A(i, j, k);
                }
            }

            Rcpp_A[k] = layer3;
        }

        print(Rcpp_sigma, nLayers_sigma, nCols_sigma, nRows_sigma);


        // Pass all three cubes to the R function
        R["mu_mat"] = Rcpp_mu;
        R["sigma_mat"] = Rcpp_sigma;
        R["A_mat"] = Rcpp_A;

        // Define a vector
        NumericVector n = NumericVector::create(2,2,2);

        // Define an integer
        int L = 3;
        int q = 2;
        // Pass the vector and integer to R
        R["n"] = n;
        R["L"] = L;
        R["q"] = q;

        R.parseEvalQ("source('mixture_dag.R')");
        R.parseEvalQ("result <- mixture_dags(mu_mat, sigma_mat, A_mat, L, q, n)");

        // Retrieve and print the result cube

        size_t rows[3], cols[3];
        rows[0]=nRows_mu;
        rows[1]=nRows_sigma;
        rows[2]=nRows_A;
        cols[0]=nCols_mu;
        cols[1]=nCols_sigma;
        cols[2]=nCols_A;

        printR(R["result"], rows, cols);
    

    return 0;
}
