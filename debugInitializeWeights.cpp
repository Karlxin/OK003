//implement debugInitializeWeights
#include "main.h"

Mat<double> debugInitializeWeights(int32_t fan_out, int32_t  fan_in)
{
	Mat<double> W;
	W.zeros(fan_out, 1 + fan_in);

	vec c = linspace<vec>(1,W.n_elem,W.n_elem);//bug fixed here

	/*cout << "c :   " << c << endl;
	system("pause");
	cout << "\n\r" << endl;*/

	W = reshape(sin(c), W.n_rows,W.n_cols) / 10;
	/*cout << "W :   " << W << endl;
	system("pause");
	cout << "\n\r" << endl;*/

	return W;
}



