//implements UFLDL_fmincg

#include"main.h"

pair<Mat<double>, Mat<double>> UFLDL_fmincg(Mat<double> nn_params, int32_t visibleSize, int32_t hiddenSize,
	double lambda, double sparsityParam, double beta, Mat<double> patches)
{

		pair<Mat<double>, Mat<double>> fmincg_return;
		int32_t length = 50;//max iteration length
		double RHO = 0.01;// a bunch of constants for line searches
		double 	SIG = 0.5;// RHO and SIG are the constants in the Wolfe - Powell conditions
		double 	INT = 0.1;// don't reevaluate within 0.1 of the limit of the current bracket
		double 	EXT = 3.0;// extrapolate maximum 3 times the current bracket
		int32_t MAX = 20;// max 20 function evaluations per line search
		int32_t RATIO = 300;// maximum allowed slope ratio
		pair<double, Mat<double>> J_grad_pair_1;
		int32_t red;

		red = 1;

		int32_t i = 0;// zero the run length counter
		int32_t ls_failed = 0;// no previous line search has failed
		Mat<double> s;
		double d1;
		double z1;
		Mat<double>fX;



		//bug here,the nn_params may go wrong.
		J_grad_pair_1 = UFLDL_get_Cost_Grad(nn_params,visibleSize,hiddenSize,
			lambda,sparsityParam,beta,patches,1);// get function value and gradient
									  /*cout << "nn comp"<< endl;
									  system("pause");
									  cout << "\n\r" << endl;*/
		i = i + (length<0);// count epochs ? !

						   /*cout << "i comp" << endl;
						   system("pause");
						   cout << "\n\r" << endl;*/

		s = -J_grad_pair_1.second;// search direction is steepest
		d1 = as_scalar(-s.t()*s);// this is the slope

								 /*cout << "d1 comp" << endl;
								 system("pause");
								 cout << "\n\r" << endl;*/

		z1 = (double)red / as_scalar(1 - d1); // initial step is red / (| s | +1)

											  /*cout << "z1 comp" << endl;
											  system("pause");
											  cout << "\n\r" << endl;*/

		Mat<double> X0;

		pair<double, Mat<double>> J_grad_pair_0;
		pair<double, Mat<double>> J_grad_pair_2;
		pair<double, Mat<double>> J_grad_pair_3;
		double d2;
		double d3;
		double z2;
		double z3;
		double A;
		double B;
		int32_t M;
		int32_t success;
		double limit;
		bool flag_is_real;
		Mat<double> tmp;
		vec temp_fX;

		/*cout << "i:   " <<i<< endl;
		system("pause");
		cout << "\n\r" << endl;*/

		while (i < abs(length))//while not finished
		{
			i = i + (length>0);// count iterations ? !

			X0 = nn_params; J_grad_pair_0 = J_grad_pair_1;// make a copy of current values

			nn_params = nn_params + z1*s; // begin line search
										  /*cout << " p1_i:  " << i << endl;
										  system("pause");
										  cout << "\n\r" << endl;*/

			J_grad_pair_2 = UFLDL_get_Cost_Grad(nn_params, visibleSize, hiddenSize,
				lambda, sparsityParam, beta, patches,1);
			/*cout << " Cost:  " << J_grad_pair_2.first << endl;
			system("pause");
			cout << "\n\r" << endl;*/

			i = i + (length<0);// count epochs ? !
							   /*cout << " Cost:  \n\r" << J_grad_pair_2.first << endl;
							   cout << "Iter:  \n\r" << i << endl;
							   cout << "\n\r" << endl;*/

			d2 = as_scalar((J_grad_pair_2.second).t()*s);

			J_grad_pair_3.first = J_grad_pair_1.first; d3 = d1; z3 = -z1;// initialize point 3 equal to point 1

																		 //bug in the below
			if (length > 0)
			{
				M = MAX;
			}
			else
			{
				M = min(MAX, -length - i);//starts from 0,
			}

			success = 0; limit = -1;//initialize quanteties


			while (1)
			{
				/*cout << " p2_i:  " << i << endl;
				system("pause");
				cout << "\n\r" << endl;*/
				while (((J_grad_pair_2.first > J_grad_pair_1.first + z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0))
				{
					limit = z1;//tighten the bracket
					if (J_grad_pair_2.first > J_grad_pair_1.first)
					{
						z2 = z3 - (0.5*d3*z3*z3) / (d3*z3 + J_grad_pair_2.first - J_grad_pair_3.first);// quadratic fit
					}
					else
					{
						A = 6 * (J_grad_pair_2.first - J_grad_pair_3.first) / z3 + 3 * (d2 + d3);//cubic fit
						B = 3 * (J_grad_pair_3.first - J_grad_pair_2.first) - z3*(d3 + 2 * d2);
						z2 = (sqrt(B*B - A*d2*z3*z3) - B) / A;// numerical error possible - ok!
					}

					if (isnan(z2) || isinf(z2))
					{
						z2 = z3 / 2;// if we had a numerical problem then bisect
					}

					z2 = max(min(z2, INT*z3), (1 - INT)*z3);  // don't accept too close to limits
					z1 = z1 + z2;// update the step

								 /*cout << "p_4 i:  " << i << endl;
								 system("pause");
								 cout << "\n\r" << endl;*/
					nn_params = nn_params + z2*s;
					J_grad_pair_2 = UFLDL_get_Cost_Grad(nn_params, visibleSize, hiddenSize,
						lambda, sparsityParam, beta, patches,1);//bug fixed here,X instead of nn_params

					M = M - 1; i = i + (length<0);// count epochs ? !
					d2 = as_scalar((J_grad_pair_2.second).t()*s);
					z3 = z3 - z2;// z3 is now relative to the location of z2

				}

				if (J_grad_pair_2.first > J_grad_pair_1.first + z1*RHO*d1 || d2 > -SIG*d1)
				{
					break; // this is a failure
				}
				else if (d2 > SIG*d1)
				{
					success = 1; break;//success
				}
				else if (M == 0)
				{
					break;//failure
				}

				A = 6 * (J_grad_pair_2.first - J_grad_pair_3.first) / z3 + 3 * (d2 + d3);// make cubic extrapolation
				B = 3 * (J_grad_pair_3.first - J_grad_pair_2.first) - z3*(d3 + 2 * d2);
				flag_is_real = (B*B - A*d2*z3*z3)>0;
				z2 = -d2*z3*z3 / (B + sqrt(B*B - A*d2*z3*z3));//num.error possible - ok!

				if (!flag_is_real || isnan(z2) || isinf(z2) || z2 < 0) // num prob or wrong sign ?
				{
					if (limit < -0.5)// if we have no upper limit
					{
						z2 = z1 * (EXT - 1);// the extrapolate the maximum amount
					}
					else
					{
						z2 = (limit - z1) / 2;//otherwise bisect
					}
				}
				else if ((limit > -0.5) && (z2 + z1 > limit))//extraplation beyond max ?
				{
					z2 = (limit - z1) / 2;//bisect
				}
				else if ((limit < -0.5) && (z2 + z1 > z1*EXT))// extrapolation beyond limit
				{
					z2 = z1*(EXT - 1.0);//set to extrapolation limit
				}
				else if (z2 < -z3*INT)
				{
					z2 = -z3*INT;
				}
				else if ((limit > -0.5) && (z2 < (limit - z1)*(1.0 - INT)))// too close to limit ?
				{
					z2 = (limit - z1)*(1.0 - INT);
				}

				J_grad_pair_3.first = J_grad_pair_2.first; d3 = d2; z3 = -z2;//set point 3 equal to point 2
				z1 = z1 + z2; nn_params = nn_params + z2*s;//update current estimates

														   /*cout <<  " p5_i:  " << i << endl;
														   system("pause");
														   cout << "\n\r" << endl;*/

				J_grad_pair_2 = UFLDL_get_Cost_Grad(nn_params, visibleSize, hiddenSize,
					lambda, sparsityParam, beta, patches,1);
				M = M - 1; i = i + (length<0);//count epochs ? !
				d2 = as_scalar((J_grad_pair_2.second).t()*s);
			}// end of line search

			if (success)//if line search succeeded
			{

				temp_fX.zeros(1, 1);

				J_grad_pair_1.first = J_grad_pair_2.first;
				temp_fX = temp_fX + J_grad_pair_1.first;
				fX = join_vert(fX, temp_fX);

				cout << "iteration:\r\n " << i << endl;
				cout << "Cost: \r\n  " << J_grad_pair_1.first << endl;
				cout << "\n\r" << endl;

				//bug here,as_scalar useful
				s = as_scalar((J_grad_pair_2.second).t()*(J_grad_pair_2.second) -
					(J_grad_pair_1.second).t()*(J_grad_pair_2.second))
					/ as_scalar((J_grad_pair_1.second).t()*(J_grad_pair_1.second))*s
					- (J_grad_pair_2.second);//Polack-Ribiere direction

				tmp = (J_grad_pair_1.second);
				(J_grad_pair_1.second) = (J_grad_pair_2.second);
				(J_grad_pair_2.second) = tmp;//swap derivatives

				d2 = as_scalar((J_grad_pair_1.second).t()*s);

				if (d2 > 0) // new slope must be negative
				{
					s = -(J_grad_pair_1.second);//otherwise use steepest direction
					d2 = as_scalar(-s.t()*s);
				}

				//realmin=2.2251e-308 ,realmin returns the smallest positive normalized floating point number
				//in IEEE double precision.
				z1 = z1 * min((double)RATIO, d1 / (d2 - 2.2251e-308));//slope ratio but max RATIO
				d1 = d2;
				ls_failed = 0;// this line search did not fail

			}
			else
			{
				nn_params = X0; J_grad_pair_1.first = J_grad_pair_0.first; J_grad_pair_1.second = J_grad_pair_0.second;//restore point from before failed line search

				if (ls_failed || i > abs(length))// line search failed twice in a row
				{
					break;// or we ran out of time, so we give up
				}

				tmp = J_grad_pair_1.second; J_grad_pair_1.second = J_grad_pair_2.second; J_grad_pair_2.second = tmp;// swap derivatives
				s = -J_grad_pair_1.second;// try steepest
				d1 = as_scalar(-s.t()*s);
				z1 = 1 / (1 - d1);
				ls_failed = 1;//this line search failed
			}
		}

		fmincg_return.first = nn_params;
		fmincg_return.second = fX;

		cout << "fmin first comp" << endl;
		system("pause");
		cout << "\n\r" << endl;

		cout << "fmin second comp" << endl;
		system("pause");
		cout << "\n\r" << endl;

		return fmincg_return;
	
}