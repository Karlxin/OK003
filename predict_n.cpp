//implement predict_n
#include "main.h"

Mat<double> predict_n(Mat<double>nn_params, Mat<double> X, Mat<uint32_t> Theta_indicator, Mat<int32_t> layer_size)
{
	Mat<double> predict_return;
	//this file has a negative subscript bug...we should find it.
	field<Mat<double>>theta_field(layer_size.n_rows - 1, 1);

	for (int i = 0; i < layer_size.n_rows - 1; i++)
	{
		theta_field(i) = (reshape(nn_params.rows(Theta_indicator(i), \
			Theta_indicator(i + 1) - 1), layer_size(i + 1), (layer_size(i) + 1)));
	}

	// Useful values
	int32_t m;//5000
	Mat<double> h1;//5000*133
	Mat<double> h2;//5000*33
	Mat<double> h3;//5000*10
	Mat<double> temp;//5000*1
	m = X.n_rows;//5000

	predict_return.zeros(X.n_rows, 1);//5000*1
	temp.ones(m, 1);//5000*1

	field<Mat<double>>h_field(layer_size.n_rows - 1, 1);
	h_field(0) = sigmoid(join_horiz(temp, X) * theta_field(0).t());

	for (int i = 1; i < layer_size.n_rows - 1; i++)
	{
		h_field(i)= sigmoid(join_horiz(temp, h_field(i-1)) * theta_field(i).t());
	}

	predict_return = conv_to<Mat<double>>::from(index_max(h_field(layer_size.n_rows-2), 1));//armadillo starts from 0,so,1 is the horiz
	predict_return++;//starts from 0

	return predict_return;
	// ======================================================================== 
}