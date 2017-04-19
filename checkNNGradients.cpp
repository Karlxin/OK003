//implement checkNNGradients
#include "main.h"

/**
* Extend division reminder to vectors
*
* @param   a       Dividend
* @param   n       Divisor
*/
template<typename T>
T mod(T a, int n)
{
	return (a - floor(a / n)*n);
}




void checkNNGradients(double lambda)
{
	int32_t input_layer_size = 3;
	int32_t hidden_layer_size = 5;
	int32_t num_labels = 3;
	int32_t m = 5;

	Mat<double> Theta1;
	Mat<double> Theta2;
	Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
	Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);

	/*cout << "Theta1 :   " << Theta1 << endl;
	system("pause");
	cout << "\n\r" << endl;

	cout << "Theta2 :   " << Theta2 << endl;
	system("pause");
	cout << "\n\r" << endl;*/

	Mat<double> X;
	Mat<double> y;

	vec c;
	c = linspace<vec>(1, m, m);//bug fix here

	X = debugInitializeWeights(m, input_layer_size - 1);

	Mat<double> temp_y;
	temp_y = mod<vec>(c, num_labels);

	y = 1 + temp_y;//bug fix here



	Mat<double> nn_params;
	nn_params = join_vert(vectorise(Theta1), vectorise(Theta2));;



	Mat<double> grad;
	grad = nnCostFunction_grad(nn_params, input_layer_size, hidden_layer_size,
		num_labels, X, y, lambda);

	Mat<double> numgrad;
	numgrad = computeNumericalGradient(nn_params, input_layer_size, hidden_layer_size,
		num_labels, X, y, lambda);//bug here


	Mat<double> diff;
	Mat<double> temp1;
	Mat<double> temp2;
	
	cout << "[numgrad grad]  :  " << join_horiz(numgrad,grad) << endl;
	system("pause");
	cout << "\n\r" << endl;

	temp1 = numgrad - grad;//this number is too big
	temp2 = numgrad + grad;

	cout << "[temp1 temp2]:   " << join_horiz(temp1,temp2) << endl;
	system("pause");
	cout << "\n\r" << endl;

	cout << "norm_karl(temp1) :   " << norm_karl(temp1) << endl;
	system("pause");
	cout << "\n\r" << endl;

	cout << "norm_karl(temp2)  :  " << norm_karl(temp2) << endl;
	system("pause");
	cout << "\n\r" << endl;

	diff = norm_karl(temp1) / norm_karl(temp2);

	
	cout << "should less than 1e-9\n\r" << endl;
	cout << "diff: " <<diff<< endl;
	system("pause");
	cout << "\n\r" << endl;
}


