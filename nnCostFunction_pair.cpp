//implement  nnCostFunction_pair
#include "main.h"

pair<double, Mat<double>>  nnCostFunction_pair(Mat<double> nn_params, int32_t input_layer_size, int32_t hidden_layer_size,
	int32_t num_labels, Mat<double> X, Mat<double> y, double lambda)
{
	pair<double, Mat<double>> J_grad_pair;
	Mat<double> grad;
	double J;

	Mat<double> Theta1;
	Mat<double> Theta2;
	/*cout << "Theta def comp." << endl;
	system("pause");
	cout << "\n\r" << endl;

	cout << nn_params.n_rows << endl;
	system("pause");
	cout << "\n\r" << endl;*/

	Theta1 = reshape(nn_params.rows(1 - 1, hidden_layer_size * (input_layer_size + 1) - 1),
		hidden_layer_size, (input_layer_size + 1));
	Theta2 = reshape(nn_params.rows((1 + (hidden_layer_size * (input_layer_size + 1))) - 1, nn_params.n_rows - 1),
		num_labels, (hidden_layer_size + 1));

	/*cout << "\n\r" << endl;
	cout << "\n\r" << endl;
	cout << "\n\r" << endl;
	cout << "\n\r" << endl;
	cout << "\n\r" << endl;

	cout << Theta1.n_rows << "   " << Theta1.n_cols << "\n\r" << endl;
	cout << Theta2.n_rows << "   " << Theta2.n_cols << "\n\r" << endl;

	cout << "Theta reshape comp." << endl;
	system("pause");
	cout << "\n\r" << endl;*/

	int32_t m;
	m = X.n_rows;
	J = 0;
	Mat<double> Theta1_grad;
	Mat<double> Theta2_grad;

	Theta1_grad.zeros(Theta1.n_rows, Theta1.n_cols);
	Theta2_grad.zeros(Theta2.n_rows, Theta2.n_cols);

	/*cout << Theta1_grad.n_rows << "   " << Theta1_grad.n_cols << "\n\r" << endl;
	cout << Theta2_grad.n_rows << "   " << Theta2_grad.n_cols << "\n\r" << endl;

	cout << "Theta grad init comp." << endl;
	system("pause");
	cout << "\n\r" << endl;*/

	vec X_vec_temp1;
	X_vec_temp1.ones(m, 1);
	X = join_horiz(X_vec_temp1, X);

	Mat<double> a1;
	Mat<double> z2;
	Mat<double> a2;
	Mat<double> z3;
	Mat<double> a3;
	Mat<double> hx;
	Mat<double> yk;

	a1 = X;
	z2 = Theta1*a1.t();
	a2 = sigmoid(z2);
	/*cout << "sum(sum(a1)):\r\n" << endl;
	cout << sum(sum(a1)) << endl;
	cout << "sum(sum(z2)):\r\n" << endl;
	cout << sum(sum(z2)) << endl;
	cout << "sum(sum(a2)):\r\n" << endl;
	cout << sum(sum(a2)) << endl;
	system("pause");
	cout << "\n\r" << endl;*/



	a2 = join_horiz(X_vec_temp1, a2.t());
	z3 = Theta2*a2.t();
	a3 = sigmoid(z3);
	/*cout << "sum(sum(a2)):\r\n" << endl;
	cout << sum(sum(a2)) << endl;
	cout << "sum(sum(z3)):\r\n" << endl;
	cout << sum(sum(z3)) << endl;
	cout << "sum(sum(a3)):\r\n" << endl;
	cout << sum(sum(a3)) << endl;
	system("pause");
	cout << "\n\r" << endl; */


	hx = sigmoid(z3);

	yk.zeros(num_labels, m);

	/*cout << "yk init comp." << endl;
	system("pause");
	cout << "\n\r" << endl;*/


	int32_t i;
	for (i = 1; i <= m; i++)
	{
		yk(y(i - 1, 0) - 1, i - 1) = 1; //we starts at 0 instead of 1
	}

	/*system("pause");
	cout << "\n\r" << endl;

	cout << "for-loop_1 comp." << endl;
	system("pause");
	cout << "\n\r" << endl;*/

	double J_temp;
	//here we use "/m" instead of "(1/m)*", this bug takes karl lots of time!
	//"1/m" this one is zero...
	//bug here again?
	
	/*cout << "sum(sum(yk)):\r\n" << endl;
	cout << sum(sum(yk)) << endl;
	cout << "sum(sum(hx)):\r\n" << endl;
	cout << sum(sum(hx)) << endl;
	system("pause");
	cout << "\n\r" << endl; */

	J_temp = (1 / (double)m)*sum(sum(((-yk) % log(hx)) - ((1 - yk) % log(1 - hx))));

	/*cout << J_temp << endl;
	cout << "J_temp comp." << endl;
	system("pause");
	cout << "\n\r" << endl;*/
	/*cout << "sum(sum(Theta1)):\r\n" << endl;
	cout << sum(sum(Theta1)) << endl;
	cout << "sum(sum(Theta2)):\r\n" << endl;
	cout << sum(sum(Theta2)) << endl;
	system("pause");
	cout << "\n\r" << endl; */

	J = J_temp + lambda / (2 * m) * (sum(sum(square(Theta1.cols(2 - 1, Theta1.n_cols - 1)))) +
		sum(sum(square(Theta2.cols(2 - 1, Theta2.n_cols - 1)))));

	J_grad_pair.first = J;

	/*cout << ("J=%f", J) << endl;
	cout << "J comp." << endl;
	system("pause");
	cout << "\n\r" << endl;*/

	Mat<double> one_temp;
	one_temp.ones(1, 1);

	for (int32_t t = 1; t <= m; t++)
	{
		a1 = X.row(t - 1);
		/*cout << "a1 comp." << endl;
		system("pause");
		cout << "\n\r" << endl; */

		z2 = Theta1 * a1.t();
		/*cout << "z2 comp." << endl;
		system("pause");
		cout << "\n\r" << endl; */

		a2 = sigmoid(z2);
		a2 = join_vert(one_temp, a2);
		/*cout << "a2 comp." << endl;
		system("pause");
		cout << "\n\r" << endl; */


		z3 = Theta2 * a2;
		/*cout << "z3 comp." << endl;
		system("pause");
		cout << "\n\r" << endl; */


		a3 = sigmoid(z3);
		/*cout << "a3 comp." << endl;
		system("pause");
		cout << "\n\r" << endl; */

		Mat<double> d3;
		Mat<double> d2;

		d3 = a3 - yk.col(t - 1);
		/*cout << "d3 comp." << endl;
		system("pause");
		cout << "\n\r" << endl;*/

		z2 = join_vert(one_temp, z2);
		/*cout << "z2 comp." << endl;
		system("pause");
		cout << "\n\r" << endl;*/

		d2 = (Theta2.t() * d3) % sigmoidGradient(z2);

		d2 = d2.rows(2 - 1, d2.n_rows - 1);
		/*cout << "d2 comp." << endl;
		system("pause");
		cout << "\n\r" << endl;



		cout << Theta2_grad.n_rows << "   " << Theta2_grad.n_cols << "\n\r" << endl;
		cout << d3.n_rows << "   " << d3.n_cols << "\n\r" << endl;
		cout << a2.n_rows << "   " << a2.n_cols << "\n\r" << endl;*/

		//error happen here
		Theta2_grad = Theta2_grad + d3 * a2.t();
		/*cout << "Theta2_grad comp." << endl;
		system("pause");
		cout << "\n\r" << endl; */

		Theta1_grad = Theta1_grad + d2 * a1;

	}
	/*cout << "grad for-loop comp." << endl;
	system("pause");
	cout << "\n\r" << endl;*/


	Theta2_grad = (1 / (double)m) * Theta2_grad;
	Theta1_grad = (1 / (double)m) * Theta1_grad;

	Theta2_grad.cols(2 - 1, Theta2_grad.n_cols - 1) = 
		Theta2_grad.cols(2 - 1, Theta2_grad.n_cols - 1) + 
		(lambda / m) *Theta2.cols(2 - 1, Theta2.n_cols - 1);//bug fix here ,Theta2 instead of Theta2_grad

	Theta1_grad.cols(2 - 1, Theta1_grad.n_cols - 1) = 
		Theta1_grad.cols(2 - 1, Theta1_grad.n_cols - 1) + 
		(lambda / m) * Theta1.cols(2 - 1, Theta1.n_cols - 1);

	grad = join_vert(vectorise(Theta1_grad), vectorise(Theta2_grad));

	J_grad_pair.second = grad;

	/*cout << "grad comp." << endl;
	system("pause");
	cout << "\n\r" << endl;*/

	return J_grad_pair;
}