//implement UFLDL_checkNNGradients

#include "main.h"

void UFLDL_checkNNGradients(Mat<double> nn_params, int32_t visibleSize, int32_t hiddenSize,
	double lambda, double sparsityParam, double beta, Mat<double> patches)
{
	
	Mat<double> numgrad;
	Mat<double> perturb;
	Mat<double> grad;
	numgrad.zeros(nn_params.n_rows, nn_params.n_cols);
	perturb.zeros(nn_params.n_rows, nn_params.n_cols);
	double e = 1e-4;
	int32_t p;
	pair<double, Mat<double>>loss1_pair;
	pair<double, Mat<double>>loss2_pair;

	loss1_pair = UFLDL_get_Cost_Grad(nn_params, visibleSize, hiddenSize,
		lambda, sparsityParam, beta, patches, 1);//first we get a analytical grad
	grad = loss1_pair.second;
	for (p = 1; p <= nn_params.n_elem; p++)
	{
		perturb(p - 1) = e;

		loss1_pair = UFLDL_get_Cost_Grad(nn_params-perturb,visibleSize,hiddenSize,
			lambda,sparsityParam,beta,patches,0);//bug inside,with the perturb change,we do not change the loss
		loss2_pair= UFLDL_get_Cost_Grad(nn_params +perturb, visibleSize, hiddenSize,
			lambda, sparsityParam, beta, patches,0);//bug inside,we nn_params equal to zero!this should not happen.

		numgrad(p - 1) = (loss2_pair.first - loss1_pair.first) / (2 * e);//bug here,from 6 and next,we get larger than correct
		perturb(p - 1) = 0;
		cout << "numgrad;grad;\n\r" << numgrad(p - 1) << "   " << grad(p - 1) << endl;
	}

	Mat<double> temp1;
	Mat<double> temp2;
	Mat<double> diff;
	temp1 = numgrad - grad;//this number is too big
	temp2 = numgrad + grad;


	diff = norm_karl(temp1) / norm_karl(temp2);

	cout << "diff:   \n\r" << diff<<endl;

	
}


