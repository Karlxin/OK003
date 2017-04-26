//implements UFLDL_get_Cost_Grad_n_show

#include "main.h"

pair<double, Mat<double>> UFLDL_get_Cost_Grad_n_show(Mat<double> theta, Mat<int32_t> layer_size,
	double lambda, double sparsityParam, double beta, Mat<double> patches, uint32_t option,
	Mat<uint32_t> Theta_indicator_UFLDL)
{
	//when option==0,just return the cost J.
	pair<double, Mat<double>> cost_grad_return;
	field<Mat<double>> W(layer_size.n_rows - 1, 1);
	field<Mat<double>> b(layer_size.n_rows - 1, 1);
	field<Mat<double>> Wgrad(layer_size.n_rows - 1, 1);
	field<Mat<double>> bgrad(layer_size.n_rows - 1, 1);
	field<Mat<double>>theta_field(layer_size.n_rows - 1, 1);

	for (int i = 0; i < layer_size.n_rows - 1; i++)
	{
		theta_field(i) = (reshape(theta.rows(Theta_indicator_UFLDL(i), \
			Theta_indicator_UFLDL(i + 1) - 1), layer_size(i + 1), (layer_size(i) + 1)));
		W(i) = theta_field(i).cols(0, theta_field(i).n_cols - 2);
		b(i) = theta_field(i).col(theta_field(i).n_cols - 1);
	}

	cost_grad_return.first = 0;


	uint32_t m = patches.n_cols;


	field<Mat<double>>a(layer_size.n_rows - 1, 1);
	field<Mat<double>>z(layer_size.n_rows - 1, 1);
	Mat<double> h;
	Mat<double> y;

	a(0) = patches;
	z(0) = W(0)*a(0) + repmat(b(0), 1, m);

	for (uint32_t i = 1; i < layer_size.n_rows - 1; i++)
	{
		a(i) = sigmoid(z(i - 1));
		z(i) = W(i)*a(i) + repmat(b(i), 1, m);
	}

	int32_t core_temp = layer_size.n_rows / 2;

	if (option == 2)//option==2,we show the core 
	{
		cout << a(core_temp).n_rows << "   " << a(core_temp).n_cols <<"\n\r"<< endl;
		cout << a(core_temp).col(3550)<< endl;

	}

	h = sigmoid(z(layer_size.n_rows - 2));
	y = a(0);

	field<Mat<double>>rho(layer_size.n_rows - 2, 1);//rho(0) correspond to a(1)
	Mat<double> squared_error;
	double regularization_term = 0;
	//Mat<double> rho;
	double sparsity_penalty = 0;
	squared_error = 0.5*sum(square((h - y)), 0);//the direction 0 stand for matlab 1 as the vertical direction

	for (uint32_t i = 0; i<layer_size.n_rows - 2; i++)
	{
		rho(i) = (1 / (double)m)*sum(a(i + 1), 1);
		sparsity_penalty += as_scalar(sum(sum(beta*((sparsityParam*log(sparsityParam / rho(i))) +
			((1 - sparsityParam)*log((1 - sparsityParam) / (1 - rho(i))))))));
		regularization_term += sum(sum(square(W(i))));
	}
	regularization_term += sum(sum(square(W(layer_size.n_rows - 2))));

	sparsity_penalty = 0;//we temporarily don't need this.
	cost_grad_return.first = as_scalar((1 / (double)m)*sum(sum(squared_error)) + (lambda / (2 * (double)m)) * regularization_term + sparsity_penalty);

	if (option == 0)//we just need cost
	{
		return cost_grad_return;
	}


	field<Mat<double>> grad_z(layer_size.n_rows - 1, 1);//3,2 map to--->(1),(0),last but one correspond to a last
	field<Mat<double>> delta(layer_size.n_rows - 1, 1);//3,2 map to--->(1),(0)
	field<Mat<double>> Delta_W(layer_size.n_rows - 1, 1);//2,1 map to--->(1),(0)
	field<Mat<double>> Delta_b(layer_size.n_rows - 1, 1);//2,1 map to--->(1),(0)

	grad_z(layer_size.n_rows - 2) = h % (1 - h);
	delta(layer_size.n_rows - 2) = -(y - h) % grad_z(layer_size.n_rows - 2);

	Delta_W(layer_size.n_rows - 2) = delta(layer_size.n_rows - 2) * a(layer_size.n_rows - 2).t();
	Delta_b(layer_size.n_rows - 2) = sum(delta(layer_size.n_rows - 2), 1);

	Wgrad(layer_size.n_rows - 2) = Delta_W(layer_size.n_rows - 2) + lambda*W(layer_size.n_rows - 2);
	bgrad(layer_size.n_rows - 2) = Delta_b(layer_size.n_rows - 2);

	//bug fix here,with inverse for loop,i need to be int instead of uint
	for (int32_t i = layer_size.n_rows - 3; i >= 0; i--)
	{
		grad_z(i) = a(i + 1) % (1 - a(i + 1));
		delta(i) = (W(i + 1).t()*delta(i + 1) +
			repmat(beta*(-sparsityParam / rho(i) +
			(1 - sparsityParam) / (1 - rho(i))), 1, m)) % grad_z(i);
		Delta_W(i) = delta(i) * a(i).t();
		Delta_b(i) = sum(delta(i), 1);
		Wgrad(i) = Delta_W(i) + lambda*W(i);
		bgrad(i) = Delta_b(i);
	}

	cost_grad_return.second = join_vert(vectorise(Wgrad(0)), vectorise(bgrad(0)));

	for (uint32_t i = 1; i < layer_size.n_rows - 1; i++)
	{
		cost_grad_return.second = join_vert(join_vert(cost_grad_return.second, vectorise(Wgrad(i))), vectorise(bgrad(i)));
	}


	cost_grad_return.second = cost_grad_return.second / (double)m;
	return cost_grad_return;

}