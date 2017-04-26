//implements fminlbfgs
#include "main.h"
//for fminlbfgs top
extern Mat<int32_t> layer_size_fminlbfgs;
//layer_size_fminlbfgs = { 400,133 };//the W1:133*400.b1:133
//layer_size_fminlbfgs = layer_size.t();
extern double sparsityParam_fminlbfgs;// desired average activation of the hidden units.
extern double lambda_fminlbfgs;// weight decay parameter
extern double beta_fminlbfgs;//weight of sparsity penalty term
extern Mat<double> patches_fminlbfgs;
extern Mat<uint32_t> Theta_indicator_UFLDL;
//for fminlbfgs bottom

static lbfgsfloatval_t evaluate(
	void *instance,
	lbfgsfloatval_t *x,
	lbfgsfloatval_t *g,
	const int n,
	const lbfgsfloatval_t step
	)
{
	int i;

	lbfgsfloatval_t fx = 0.0;
	vec vtemp_x(x, n, true, false);
	vec vtemp_g(g, n, false, false);
	pair<double, Mat<double>> J_grad_pair;

	J_grad_pair = UFLDL_get_Cost_Grad_n(vtemp_x, layer_size_fminlbfgs,
		lambda_fminlbfgs, sparsityParam_fminlbfgs, beta_fminlbfgs, patches_fminlbfgs, 1, Theta_indicator_UFLDL);

	vtemp_g = J_grad_pair.second;

	return (lbfgsfloatval_t)J_grad_pair.first;
}

static int progress(
	void *instance,
	const lbfgsfloatval_t *x,
	const lbfgsfloatval_t *g,
	const lbfgsfloatval_t fx,
	const lbfgsfloatval_t xnorm,
	const lbfgsfloatval_t gnorm,
	const lbfgsfloatval_t step,
	int n,
	int k,
	int ls
)
{
	printf("Iteration %d:\n", k);
	printf("  fx = %f \n", fx);
	printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
	printf("\n");
	return 0;
}


Col<double> fminlbfgs(Col<double> x_init,uint32_t N)
{
	Col<double> x_return;
	int i, ret = 0;
	lbfgsfloatval_t fx;
	lbfgsfloatval_t *x = lbfgs_malloc(N);
	lbfgs_parameter_t param;

	if (x == NULL) {
		printf("ERROR: Failed to allocate a memory block for variables.\n");
		x_return = { 1 };
		return x_return;
	}

	x= x_init.memptr();

	/* Initialize the parameters for the L-BFGS optimization. */
	lbfgs_parameter_init(&param);
	/*param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;*/

	/*
	Start the L-BFGS optimization; this will invoke the callback functions
	evaluate() and progress() when necessary.
	*/
	ret = lbfgs(N, x, &fx, evaluate, progress, NULL, &param);

	/* Report the result. */
	printf("L-BFGS optimization terminated with status code = %d\n", ret);
	printf("  fx = %f \n", fx);
	system("pause");

	x_return = x_init;
	//lbfgs_free(x);//this line will cause breakout.


	return x_return;
}
