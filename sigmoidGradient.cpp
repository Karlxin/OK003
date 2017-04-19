//implement sigmoidGradient
#include "main.h"

Mat<double> sigmoidGradient(Mat<double> X)
{
	Mat<double> g;
	g.zeros(X.n_rows, X.n_cols);

	g = exp(-X) /square(exp(-X) + 1);
	return g;
}