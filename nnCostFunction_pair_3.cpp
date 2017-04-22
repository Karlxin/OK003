//Implements the neural network cost function for a three layer
#include "main.h"

pair<double, Mat<double>>  nnCostFunction_pair_3(Mat<double> nn_params, 
	Mat<int32_t> layer_size, Mat<double> X, Mat<double> y,
	double lambda, Mat<uint32_t> Theta_indicator, int32_t choice)
{
	//remember we starts at 0
	pair<double, Mat<double>> J_grad_pair;
#define Theta(k) (reshape(nn_params.rows(Theta_indicator(k),\
 Theta_indicator(k+1)-1),layer_size(k+1), (layer_size(k) + 1)))

	Mat<double> Theta1;
	Mat<double> Theta2;
	Mat<double> Theta3;
	Theta1 = Theta(0);
	Theta2 = Theta(1);
	Theta3 = Theta(2);

	int32_t m;
	m = X.n_rows;
	J_grad_pair.first = 0;
	J_grad_pair.second.zeros(nn_params.n_rows);

	vec X_vec_temp1;
	X_vec_temp1.ones(m, 1);
	X = join_horiz(X_vec_temp1, X);//now we get a bias term

	Mat<double> a1;//5000*401
	Mat<double> a2;//133*5000
	Mat<double> a3;//33*5000
	Mat<double> z2;//133*5000
	Mat<double> z3;//33*5000
	Mat<double> z4;//10*5000

	Mat<double> hx;//10*5000
	Mat<double> yk;//10*5000

	a1 = X;//5000*401

	z2 = Theta1 * a1.t();//133*401,401*5000;133*5000
	a2 = sigmoid(z2);//133*5000
	a2 = join_vert(X_vec_temp1.t(), a2);//134*5000

	z3 = Theta2 * a2;//33*134,134*5000;33*5000
	a3 = sigmoid(z3);//33*5000
	a3 = join_vert(X_vec_temp1.t(), a3);//34*5000

	z4 = Theta3*a3;//10*34,34*5000;10*5000

	hx = sigmoid(z4);//10*5000
	yk.zeros(layer_size(layer_size.n_rows - 1), m);//10*5000

	for (int32_t i = 1; i <= m; i++)
	{
		yk(y(i - 1, 0) - 1, i - 1) = 1; //we starts at 0 instead of 1
	}

	double J_temp;

	J_temp = (1 / (double)m)*sum(sum(((-yk) % log(hx)) - ((1 - yk) % log(1 - hx))));

	J_grad_pair.first = J_temp + lambda / (2 * m) * (sum(sum(square(nn_params))));

	if (choice = 0)//we just need J
	{
		return J_grad_pair;
	}

	Mat<double> d4;
	Mat<double> d3;
	Mat<double> d2;
	Mat<double> Theta1_grad;
	Mat<double> Theta2_grad;
	Mat<double> Theta3_grad;
	//vectorize this top
	d4 = hx - yk;//10*5000

	z3 = join_vert(X_vec_temp1.t(), z3);//34*5000
	d3 = (Theta3.t() * d4) % sigmoidGradient(z3);//34*5000
	d3 = d3.rows(2 - 1, d3.n_rows - 1);//33*5000

	z2 = join_vert(X_vec_temp1.t(), z2);//134*5000
	d2 = (Theta2.t() * d3) % sigmoidGradient(z2);//134*5000
	d2 = d2.rows(2 - 1, d2.n_rows - 1);//133*5000

	Theta1_grad = d2 * a1;//133*5000,5000*401;133*401;
	Theta2_grad = d3 * a2.t();//33*5000,5000*134;33*134;
	Theta3_grad = d4 * a3.t();//10*5000,5000*34;10*34;

	Theta1_grad.cols(1, Theta1_grad.n_cols - 1) =
		Theta1_grad.cols(1, Theta1_grad.n_cols - 1) +
		(lambda)* Theta1.cols(1, Theta1.n_cols - 1);

	Theta2_grad.cols(1, Theta2_grad.n_cols - 1) =
		Theta2_grad.cols(1, Theta2_grad.n_cols - 1) +
		(lambda)* Theta2.cols(1, Theta2.n_cols - 1);

	Theta3_grad.cols(1, Theta3_grad.n_cols - 1) =
		Theta3_grad.cols(1, Theta3_grad.n_cols - 1) +
		(lambda)* Theta3.cols(1, Theta3.n_cols - 1);

	J_grad_pair.second.rows(0, Theta_indicator(1) - 1) = vectorise(Theta1_grad);//Theta1_grad =d2 * a1;//133*5000,5000*401;133*401;
	J_grad_pair.second.rows(Theta_indicator(1), Theta_indicator(2) - 1) = vectorise(Theta2_grad);//Theta2_grad =d3 * a2.t();//33*5000,5000*134;33*134;
	J_grad_pair.second.rows(Theta_indicator(2), Theta_indicator(3) - 1) = vectorise(Theta3_grad);//Theta3_grad =d4 * a3.t();//10*5000,5000*34;10*34;

	J_grad_pair.second = (1 / (double)m)*J_grad_pair.second;
	//vectorize this bottom

	return J_grad_pair;
}