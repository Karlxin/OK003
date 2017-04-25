#pragma once
#include <iostream>
#include <armadillo>
#include <lbfgs.h>

using namespace std;
using namespace arma;

double nnCostFunction(Mat<double> nn_params, int32_t input_layer_size, int32_t hidden_layer_size, 
	int32_t num_labels, Mat<double> X, Mat<double> y, double lambda);

Mat<double> sigmoid(Mat<double> X);

Mat<double> sigmoidGradient(Mat<double> X);

Mat<double> randInitializeWeights(int32_t L_in, int32_t L_out);

Mat<double> debugInitializeWeights(int32_t fan_out, int32_t  fan_in);

void checkNNGradients(double lambda);

Mat<double> nnCostFunction_grad(Mat<double> nn_params, int32_t input_layer_size, int32_t hidden_layer_size,
	int32_t num_labels, Mat<double> X, Mat<double> y, double lambda);

Mat<double> computeNumericalGradient(Mat<double> nn_params, int32_t input_layer_size, int32_t hidden_layer_size,
	int32_t num_labels, Mat<double> X, Mat<double> y, double lambda);

pair<double, Mat<double>>  nnCostFunction_pair(Mat<double> nn_params, 
	int32_t input_layer_size, int32_t hidden_layer_size,
	int32_t num_labels, Mat<double> X, Mat<double> y, double lambda);

pair<Mat<double>, Mat<double>> fmincg(Mat<double> nn_params, int32_t input_layer_size, int32_t hidden_layer_size,
	int32_t num_labels, Mat<double> X, Mat<double> y, double lambda);

Mat<double> predict(Mat<double> Theta1, Mat<double> Theta2, Mat<double> X);

pair<double,Mat<double>>  nnCostFunction_pair_3(Mat<double> nn_params, Mat<int32_t> layer_size,
	Mat<double> X, Mat<double> y, double lambda, Mat<uint32_t> Theta_indicator,int32_t choice);

pair<Mat<double>, Mat<double>> fmincg_3(Mat<double> nn_params, Mat<int32_t> layer_size,
	Mat<double> X, Mat<double> y, double lambda, Mat<uint32_t> Theta_indicator);

Mat<double> predict_3(Mat<double>Theta, Mat<double> X, Mat<uint32_t> Theta_indicator, Mat<int32_t> layer_size);

pair<double, Mat<double>>  nnCostFunction_pair_n(Mat<double> nn_params, Mat<int32_t> layer_size,
	Mat<double> X, Mat<double> y, double lambda, Mat<uint32_t> Theta_indicator, int32_t choice);

pair<Mat<double>, Mat<double>> fmincg_n(Mat<double> nn_params, Mat<int32_t> layer_size,
	Mat<double> X, Mat<double> y, double lambda, Mat<uint32_t> Theta_indicator);

Mat<double> predict_n(Mat<double>Theta, Mat<double> X, Mat<uint32_t> Theta_indicator, Mat<int32_t> layer_size);

Mat<double> UFLDL_init_rand(int32_t visibleSize,int32_t hiddenSize);

pair<double, Mat<double>> UFLDL_get_Cost_Grad(Mat<double> theta, int32_t visibleSize, int32_t hiddenSize,
	double lambda,double sparsityParam, double beta, Mat<double> patches, uint32_t option);

pair<Mat<double>, Mat<double>> UFLDL_fmincg(Mat<double> theta, int32_t visibleSize, int32_t hiddenSize,
	double lambda, double sparsityParam, double beta, Mat<double> patches);

void UFLDL_checkNNGradients(Mat<double> theta, int32_t visibleSize, int32_t hiddenSize,
	double lambda, double sparsityParam, double beta, Mat<double> patches);

Col<double> fminlbfgs(Col<double> x_init, uint32_t N);


/**
* Extend 2-norm of a vector
*
* @param   a vector
*/
template<typename T>
T norm_karl(T a)
{
	return sqrt(sum(square(a)));
}

template<class Matrix>
void print_matrix(Matrix matrix) {
	matrix.print(std::cout);
}

