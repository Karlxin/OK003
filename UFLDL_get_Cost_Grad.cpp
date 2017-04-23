//implements UFLDL_get_Cost_Grad;

#include "main.h"

pair<double, Mat<double>> UFLDL_get_Cost_Grad(Mat<double> theta, int32_t visibleSize, int32_t hiddenSize,
	double lambda, double sparsityParam, double beta, Mat<double> patches,uint32_t option)
{
	pair<double, Mat<double>> cost_grad_return;
	Mat<double> W1;
	Mat<double> W2;
	Mat<double> b1;
	Mat<double> b2;
	Mat<double> W1grad;
	Mat<double> W2grad;
	Mat<double> b1grad;
	Mat<double> b2grad;

	W1 = reshape(theta.rows(0,hiddenSize*visibleSize-1), hiddenSize, visibleSize);
	W2 = reshape(theta.rows(hiddenSize*visibleSize,2 * hiddenSize*visibleSize-1), visibleSize, hiddenSize);
	b1 = theta.rows(2 * hiddenSize*visibleSize ,2 * hiddenSize*visibleSize + hiddenSize-1);
	b2 = theta.rows(2 * hiddenSize*visibleSize + hiddenSize,theta.n_rows-1);

	cost_grad_return.first = 0;
	
	W1grad.zeros(W1.n_rows, W1.n_cols);
	W2grad.zeros(W2.n_rows, W2.n_cols);
	b1grad.zeros(b1.n_rows, b1.n_cols);
	b2grad.zeros(b2.n_rows, b2.n_cols);

	uint32_t m = patches.n_cols;


	Mat<double> x;
	Mat<double> a1;
	Mat<double> a2;
	Mat<double> z2;
	Mat<double> z3;
	Mat<double> h;
	Mat<double> y;

	x = patches;
	a1 = x;
	z2 = W1*a1 + repmat(b1, 1, m);
	a2 = sigmoid(z2);
	z3 = W2*a2 + repmat(b2, 1, m);

	h= sigmoid(z3);
	y = x;


	Mat<double> squared_error;
	Mat<double> rho;
	double sparsity_penalty;
	squared_error = 0.5*sum(square((h - y)), 0);//the direction 0 stand for matlab 1 as the vertical direction
	rho = (1 / (double)m)*sum(a2, 1);

	sparsity_penalty = as_scalar(beta*sum(sparsityParam*log(sparsityParam/ rho) + (1 - sparsityParam)*log((1 - sparsityParam)/ (1 - rho))));

	cost_grad_return.first = as_scalar((1 / (double)m)*sum(squared_error,1)) + lambda / 2 * (sum(sum(square(W1))) + sum(sum(square(W2)))) + sparsity_penalty;

	if (option == 0)//we just need cost
	{
		return cost_grad_return;
	}

	Mat<double> grad_z3;
	Mat<double> delta_3;
	Mat<double> grad_z2;
	Mat<double> delta_2;
	Mat<double> Delta_W2;
	Mat<double> Delta_b2;
	Mat<double> Delta_W1;
	Mat<double> Delta_b1;

	grad_z3 = h%(1 - h);
	delta_3 = -(y - h)%grad_z3;

	grad_z2 = a2%(1 - a2);
	delta_2 = (W2.t()*delta_3+repmat(beta*(-sparsityParam/rho+(1-sparsityParam)/(1-rho)),1,m))%grad_z2;

	Delta_W2 = delta_3*a2.t();
	Delta_b2 = sum(delta_3, 1);
	Delta_W1 = delta_2*a1.t();
	Delta_b1 = sum(delta_2, 1);

	W1grad = (1 / (double)m)*Delta_W1 + lambda*W1;
	W2grad = (1 / (double)m)*Delta_W2 + lambda*W2;
	b1grad = (1 / (double)m)*Delta_b1;
	b2grad = (1 / (double)m)*Delta_b2;

	cost_grad_return.second = join_vert(join_vert(join_vert(vectorise(W1grad), 
		vectorise(W2grad)), vectorise(b1grad)), vectorise(b2grad));

	return cost_grad_return;
}