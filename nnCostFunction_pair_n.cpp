//Implements the neural network cost function for a n layer
#include "main.h"

pair<double, Mat<double>>  nnCostFunction_pair_n(Mat<double> nn_params,
	Mat<int32_t> layer_size, Mat<double> X, Mat<double> y,
	double lambda, Mat<uint32_t> Theta_indicator, int32_t choice)
{
	pair<double, Mat<double>> J_grad_pair;

	field<Mat<double>>theta_field(layer_size.n_rows - 1, 1);

	for (int i = 0; i < layer_size.n_rows - 1; i++)
	{
		theta_field(i) = (reshape(nn_params.rows(Theta_indicator(i), \
			Theta_indicator(i + 1) - 1), layer_size(i + 1), (layer_size(i) + 1)));
	}

	int32_t m;
	m = X.n_rows;
	J_grad_pair.first = 0;
	J_grad_pair.second.zeros(nn_params.n_rows);

	vec X_vec_temp1;
	X_vec_temp1.ones(m, 1);
	X = join_horiz(X_vec_temp1, X);//now we get a bias term

	field<Mat<double>>a_field(layer_size.n_rows - 1, 1);//a use the input to last but 1 layer;a_field(0) stand for a1
	field<Mat<double>>z_field(layer_size.n_rows - 1, 1);//z use the second to last layer;z_field(0) stand for z2

	Mat<double> hx;//10*5000
	Mat<double> yk;//10*5000

	a_field(0) = X;//5000*401
	z_field(0) = theta_field(0)*a_field(0).t();//133*401,401*5000;133*5000.
	for (int i = 1; i < layer_size.n_rows - 1; i++)
	{
		a_field(i) = sigmoid(z_field(i-1));//133*5000
		a_field(i)= join_vert(X_vec_temp1.t(), a_field(i));//134*5000
		z_field(i) = theta_field(i)*a_field(i);//133*401,401*5000;133*5000.
	}
	//z_field(layer_size.n_rows - 2) = theta_field(layer_size.n_rows - 2)*a_field(layer_size.n_rows - 2);

	hx = sigmoid(z_field(layer_size.n_rows - 2));
	yk.zeros(layer_size(layer_size.n_rows - 1), m);

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

	field<Mat<double>>d_field(layer_size.n_rows - 1, 1);//d_filed(0) stand for d2
	field<Mat<double>>theta_grad_field(layer_size.n_rows - 1, 1);//theta_grad_field(0) stand for theta_grad1

	d_field(layer_size.n_rows - 2) = hx - yk;
	theta_grad_field(layer_size.n_rows - 2) = d_field(layer_size.n_rows - 2)*a_field(layer_size.n_rows - 2).t();
	for (int i = layer_size.n_rows - 3; i > 0; i--)
	{
		z_field(i)= join_vert(X_vec_temp1.t(), z_field(i));
		d_field(i)= (theta_field(i+1).t() * d_field(i+1)) % sigmoidGradient(z_field(i));
		d_field(i) = d_field(i).rows(1, d_field(i).n_rows - 1);
		theta_grad_field(i) = d_field(i)*a_field(i).t();
		theta_grad_field(i).cols(1, theta_field(i).n_cols - 1) =
			theta_grad_field(i).cols(1, theta_field(i).n_cols - 1) 
			+ (lambda)*theta_grad_field(i).cols(1, theta_field(i).n_cols - 1);
	}
	z_field(0) = join_vert(X_vec_temp1.t(), z_field(0));
	d_field(0) = (theta_field(1).t() * d_field(1)) % sigmoidGradient(z_field(0));
	d_field(0) = d_field(0).rows(1, d_field(0).n_rows - 1);
	theta_grad_field(0) = d_field(0)*a_field(0);
	theta_grad_field(0).cols(1, theta_field(0).n_cols - 1) =
		theta_grad_field(0).cols(1, theta_field(0).n_cols - 1)
		+ (lambda)*theta_grad_field(0).cols(1, theta_field(0).n_cols - 1);

	for (int i = 0; i < layer_size.n_rows - 1; i++)
	{
		J_grad_pair.second.rows(Theta_indicator(i), Theta_indicator(i+1) - 1) = vectorise(theta_grad_field(i));
	}

	J_grad_pair.second = (1 / (double)m)*J_grad_pair.second;
	//vectorize this bottom

	return J_grad_pair;
}