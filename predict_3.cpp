//implement predict_3
#include "main.h"

Mat<double> predict_3(Mat<double>nn_params, Mat<double> X, Mat<uint32_t> Theta_indicator, Mat<int32_t> layer_size)
{
	Mat<double> predict_return;
	//this file has a negative subscript bug...we should find it.
#define Theta(k) (reshape(nn_params.rows(Theta_indicator(k),\
 Theta_indicator(k+1)-1),layer_size(k+1), (layer_size(k) + 1)))

	Mat<double> Theta1;
	Mat<double> Theta2;
	Mat<double> Theta3;
	Theta1 = Theta(0);
	Theta2 = Theta(1);
	Theta3 = Theta(2);

	// Useful values
	int32_t m;//5000
	int32_t num_labels;//10
	Mat<double> h1;//5000*133
	Mat<double> h2;//5000*33
	Mat<double> h3;//5000*10
	Mat<double> temp;//5000*1
	m = X.n_rows;//5000
	num_labels = Theta3.n_rows;//10

	predict_return.zeros(X.n_rows, 1);//5000*1
	temp.ones(m, 1);//5000*1

	h1 = sigmoid(join_horiz(temp, X) * Theta1.t());//5000*1 add to 5000*400 in horiz;5000*401,401*133;5000*133

	h2 = sigmoid(join_horiz(temp, h1) * Theta2.t());//5000*134,134*33;5000*33

	h3 = sigmoid(join_horiz(temp, h2) * Theta3.t());//5000*34,34*10,5000*10

	predict_return = conv_to<Mat<double>>::from(index_max(h3, 1));//armadillo starts from 0,so,1 is the horiz
	predict_return++;//starts from 0

	return predict_return;
	// ======================================================================== 
}