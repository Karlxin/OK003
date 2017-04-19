//implement randInitializeWeights
#include "main.h"

Mat<double> randInitializeWeights(int32_t L_in, int32_t L_out)
{
	Mat<double> W;
	W.zeros(L_out,1+L_in);

	double epsilon_init = 0.12;

	W = randu(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

	return W;
}
