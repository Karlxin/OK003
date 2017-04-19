//Implements computeNumericalGradient
#include "main.h"

Mat<double> computeNumericalGradient(Mat<double> nn_params, int32_t input_layer_size, int32_t hidden_layer_size,
	int32_t num_labels, Mat<double> X, Mat<double> y, double lambda)
{
	Mat<double> numgrad;
	Mat<double> perturb;
	numgrad.zeros(nn_params.n_rows, nn_params.n_cols);
	perturb.zeros(nn_params.n_rows, nn_params.n_cols);
	double e = 1e-4;
	int32_t p;
	double loss1, loss2;

	for (p = 1; p <= nn_params.n_elem; p++)
	{
		perturb(p-1) = e;//bug find here,the other element missing
		/*cout << "perturb :   " << perturb << endl;
		system("pause");
		cout << "\n\r" << endl;

		cout << "nn_params :   " << nn_params << endl;
		system("pause");
		cout << "\n\r" << endl;*/

		loss1 = nnCostFunction(nn_params-perturb,input_layer_size,hidden_layer_size,
			num_labels, X, y, lambda);//bug inside,with the perturb change,we do not change the loss
		loss2 = nnCostFunction(nn_params+perturb, input_layer_size, hidden_layer_size,
			num_labels, X, y, lambda);//bug inside,we nn_params equal to zero!this should not happen.

		/*cout << "loss2 :   " << loss2 << endl;
		system("pause");
		cout << "\n\r" << endl;

		cout << "loss1  :  " << loss1 << endl;
		system("pause");
		cout << "\n\r" << endl;

		cout << "loss2-loss1  :  " << loss2-loss1 << endl;
		system("pause");
		cout << "\n\r" << endl;*/

		numgrad(p-1) = (loss2 - loss1) / (2 * e);//bug here,from 6 and next,we get larger than correct

		/*cout << "numgrad:  " << numgrad << endl;
		system("pause");
		cout << "\n\r" << endl;*/

		perturb(p-1) = 0;
	}
	return numgrad;
}