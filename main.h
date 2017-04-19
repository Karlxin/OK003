#pragma once
#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

double nnCostFunction(Mat<double> nn_params, int32_t input_layer_size, int32_t hidden_layer_size, \
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

pair<double, Mat<double>>  nnCostFunction_pair(Mat<double> nn_params, int32_t input_layer_size, int32_t hidden_layer_size,
	int32_t num_labels, Mat<double> X, Mat<double> y, double lambda);

pair<Mat<double>, Mat<double>> fmincg(Mat<double> nn_params, int32_t input_layer_size, int32_t hidden_layer_size,
	int32_t num_labels, Mat<double> X, Mat<double> y, double lambda);

Mat<double> predict(Mat<double> Theta1, Mat<double> Theta2, Mat<double> X);

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



