//implement sigmoid function
#include "main.h"

Mat<double> sigmoid(Mat<double> X)
{
	Mat<double> g;
	g=  1.0 / (1.0 + exp(-X));
	return g;
}