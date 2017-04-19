//implement predict
#include "main.h"

Mat<double> predict(Mat<double> Theta1, Mat<double> Theta2, Mat<double> X)
{
	Mat<double> predict_return;
	//PREDICT Predict the label of an input given a trained neural network
	//   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
	//   trained weights of a neural network(Theta1, Theta2)

	// Useful values
	int32_t m;
	int32_t num_labels;
	Mat<double> h1;
	Mat<double> h2;
	Mat<double> temp;
	m = X.n_rows;
	num_labels =Theta2.n_rows;

	predict_return.zeros(X.n_rows, 1);
	temp.ones(m, 1);

	h1 = sigmoid(join_horiz(temp, X) * Theta1.t());

	//bug in below
	h2 = sigmoid(join_horiz(temp, h1) * Theta2.t());

	/*cout << "sum(sum(index_max(h2, 1))).\n\r" << endl;
	cout << sum(sum(index_max(h2, 1))) << endl;
	system("pause");
	cout << "\n\r" << endl;*/

	//bug here
	//bug fix here ,indices instead of actual value,i.e.,index_max instead of max
	//there are still bugs.
	predict_return = conv_to<Mat<double>>::from(index_max(h2, 1));//armadillo starts from 0,so,1 is the horiz
	predict_return++;

	//bug here sumsum should be 27595, we got a 27546
	cout << "sum(sum(predict_return)).\n\r" << endl;
	cout << sum(sum(predict_return)) << endl;
	system("pause");
	cout << "\n\r" << endl;

	return predict_return;
	// ======================================================================== =
}