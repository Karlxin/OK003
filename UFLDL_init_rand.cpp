//implements UFLDL_init_rand;

#include "main.h"

Mat<double> UFLDL_init_rand(int32_t visibleSize, int32_t hiddenSize)
{
	Mat<double> theta_return;
	double r;
	Mat<double> W1;
	Mat<double> W2;
	Mat<double> b1;
	Mat<double> b2;
	// Initialize parameters randomly based on layer sizes.
	r = std::sqrt((double)6) / sqrt(hiddenSize + visibleSize + 1);   // we'll choose weights uniformly from the interval [-r, r]

	W1 = W1.randu(hiddenSize, visibleSize) * 2 * r - r;
	W2 = W2.randu(visibleSize, hiddenSize) * 2 * r - r;

	b1.zeros(hiddenSize, 1);
	b2.zeros(visibleSize, 1);

	// Convert weights and bias gradients to the vector form.
	// This step will "unroll" (flatten and concatenate together) all
	// your parameters into a vector, which can then be used with minFunc.
	theta_return = join_vert(join_vert(join_vert(vectorise(W1), vectorise(W2)), vectorise(b1)), vectorise(b2));
	return theta_return;
}