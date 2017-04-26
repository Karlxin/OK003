//implements UFLDL_init_rand_n

#include "main.h"

Mat<double> UFLDL_init_rand_n(Mat<int32_t> layer_size)
{
		Mat<double> theta_return;
		double r;
		field<Mat<double>> W(layer_size.n_rows-1,1);

		field<Mat<double>> b(layer_size.n_rows-1,1);

		// Initialize parameters randomly based on layer sizes.
		r = sqrt((double)6) / sqrt(sum(sum(layer_size))+1);   // we'll choose weights uniformly from the interval [-r, r]

		W(0)=W(0).randu(layer_size(1), layer_size(0)) * 2 * r - r;
		b(0).zeros(layer_size(1),1);
		theta_return = join_vert(vectorise(W(0)), vectorise(b(0)));

		for (uint32_t i = 1; i < layer_size.n_rows - 1; i++)
		{
			W(i) = W(i).randu(layer_size(i+1), layer_size(i)) * 2 * r - r;
			b(i).zeros(layer_size(i+1), 1);
			theta_return = join_vert(join_vert(theta_return, vectorise(W(i))), vectorise(b(i)));
		}

		// Convert weights and bias gradients to the vector form.
		// This step will "unroll" (flatten and concatenate together) all
		// your parameters into a vector, which can then be used with minFunc.

		return theta_return;
	
}