#include "main.h"

// Extract the data as an Armadillo matrix Mat of type T, if there is no data the matrix will be empty
template<typename T>
arma::Mat<T> load_mat(std::ifstream &file, const std::string &keyword) {
	std::string line;
	std::stringstream ss;
	bool process_data = false;
	bool has_data = false;
	while (std::getline(file, line)) {
		if (line.find(keyword) != std::string::npos) {
			process_data = !process_data;
			if (process_data == false) break;
			continue;

		}
		if (process_data) {
			ss << line << '\n';
			has_data = true;

		}

	}

	arma::Mat<T> val;
	if (has_data) {
		val.load(ss);

	}
	return val;

}


int
main(int argc, char** argv)
{
	/*//practice_1_readMatrixFromMatlab top
	Mat<double> fromMatlab_1;

	fromMatlab_1.load("matlab_data.txt");

	cout << fromMatlab_1.n_rows <<"   "<< fromMatlab_1.n_cols <<"\n\r"<<fromMatlab_1<< endl;

	system("pause");
	//practice_1_readMatrixFromMatlab bottom*/



	//practice_2_buildNeuralNetwork top
	//neuron:input_400,middle1_25,output_10
	int32_t input_layer_size = 400;
	int32_t hidden_layer_size = 25;
	int32_t num_labels = 10;

	Mat<double> X;//the photo,20*20 pixel 5000 photos,we uniform these into float 1:0,size:5000*400
	Mat<double> y;//the label, we are about to recognize the arabic digits from integer 9:0,size:5000*1
	Mat<double> Theta1;//the weight from input to middle 1,size:25*401
	Mat<double> Theta2;//the weight from middle 1 to output,size:10*26
	Mat<double> nn_params;

	std::ifstream file("matlab_data_v2.txt");

	X= load_mat<double>(file, "X");
	y = load_mat<double>(file, "y");
	Theta1 = load_mat<double>(file, "Theta1");
	Theta2 = load_mat<double>(file, "Theta2");

	//std::ifstream file("test.mat");
	//X.load(file);
	//X.load("test.mat");
	//X.load("matlab_data_1.txt");
	//y.load("matlab_data_2.txt");
	//Theta1.load("matlab_data_3.txt");
	//Theta2.load("matlab_data_4.txt");
	



	cout << X.n_rows << "   " << X.n_cols << "\n\r" << endl;
	cout << y.n_rows << "   " << y.n_cols << "\n\r" << endl;
	cout << Theta1.n_rows << "   " << Theta1.n_cols << "\n\r" << endl;
	cout << Theta2.n_rows << "   " << Theta2.n_cols << "\n\r" << endl;

	cout << "data reading comp." << endl;
	system("pause");
	cout << "\n\r" << endl;

	nn_params = join_vert(vectorise(Theta1), vectorise(Theta2));

	cout << "nn_params reading comp." << endl;
	system("pause");
	cout << "\n\r" << endl;

	double lambda = 0;

	cout << "lambda reading comp." << endl;
	system("pause");
	cout << "\n\r" << endl;

	double J;
	J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
		num_labels, X, y, lambda);

	cout << "this value should be about 0.287629\n\r" << endl;
	cout << J << endl;
	system("pause");
	cout << "\n\r" << endl;

	lambda = 1;
	J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
		num_labels, X, y, lambda);
	cout << "this value should be about 0.383770\n\r" << endl;
	cout << J << endl;
	system("pause");
	cout << "\n\r" << endl;

	Mat<double> g;
	g << 1 << -0.5 << 0 << 0.5 << 1 << endr;
	g = sigmoidGradient(g);
	cout << "Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n\r" << endl;
	cout << g << endl;
	system("pause");
	cout << "\n\r" << endl;



	Mat<double> initial_Theta1;
	Mat<double> initial_Theta2;
	Mat<double> initial_nn_params;
	initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
	initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
	initial_nn_params = join_vert(vectorise(initial_Theta1), vectorise(initial_Theta2));

	cout << "initial_params comp." << endl;
	system("pause");
	cout << "\n\r" << endl;

	lambda = 0;

	checkNNGradients(lambda);

	cout << "checkNNGradients comp." << endl;
	system("pause");
	cout << "\n\r" << endl;

	lambda = 3;
	checkNNGradients(lambda);

	cout << "checkNNGradients change lambda comp." << endl;
	system("pause");
	cout << "\n\r" << endl;


	double debug_J;
	debug_J = nnCostFunction(nn_params, input_layer_size,
		hidden_layer_size, num_labels, X, y, lambda);

	cout << "this value should be about 0.576051.\n\r" << endl;
	cout << "debug_J=   " << debug_J << endl;
	system("pause");
	cout << "\n\r" << endl;

	cout << "Training Neural Network... " << endl;
	system("pause");
	cout << "\n\r" << endl;

	lambda = 1;//we missed this line.

	//TODO:fix the fmincg,let it fast!although it can decrease the cost.compare to the matlab
	pair<Mat<double>, Mat<double>> fmincg_temp;

	//bug fix here. initial_nn_params instead of nn_params.the initial value matter!
	fmincg_temp = fmincg(initial_nn_params, input_layer_size, hidden_layer_size,
		num_labels, X, y, lambda);

	cout << "fmincg done... " << endl;
	system("pause");
	cout << "\n\r" << endl;

	cout << fmincg_temp.first.n_rows << "   " << fmincg_temp.first.n_cols << "\n\r" << endl;
	cout << fmincg_temp.first.n_rows << "   " << fmincg_temp.first.n_cols << "\n\r" << endl;

	Theta1 = reshape((fmincg_temp.first).rows(1 - 1, hidden_layer_size * (input_layer_size + 1) - 1),
		hidden_layer_size, (input_layer_size + 1));
	Theta2 = reshape((fmincg_temp.first).rows((1 + (hidden_layer_size * (input_layer_size + 1))) - 1, (fmincg_temp.first).n_rows - 1),
		num_labels, (hidden_layer_size + 1));

	/*cout << "sum(sum(Theta1)):\r\n" << endl;
	cout << sum(sum(Theta1)) << endl;
	cout << "sum(sum(Theta2)):\r\n" << endl;
	cout << sum(sum(Theta2)) << endl;
	system("pause");
	cout << "\n\r" << endl;*/

	Mat<double> pred;
	//bug inside predict
	pred = predict(Theta1, Theta2, X);

	/*cout << "y.rows(0, 25) :\r\n" << endl;
	cout << y.rows(0,25) << endl;
	cout << "pred.rows(0, 25) :\r\n" << endl;
	cout << pred.rows(0, 25) << endl;
	cout << "pred==y:\r\n" << endl;
	cout << conv_to<Mat<double>>::from(pred==y) << endl;
	cout << "(mean(conv_to<Mat<double>>::from(pred==y))):\r\n" << endl;
	cout << (mean(conv_to<Mat<double>>::from(pred == y))) << endl;
	cout << "(mean(conv_to<Mat<double>>::from(pred==y))) * 100:\r\n" << endl;
	cout << (mean(conv_to<Mat<double>>::from(pred == y))) * 100 << endl;
	system("pause");
	cout << "\n\r" << endl;*/

	//bug fix here.mean of a binary return also binary!!! so should change to double!!!
	cout << "\nTraining Set Accuracy:    \n " << (mean(conv_to<Mat<double>>::from(pred == y))) * 100 << endl;//bug here
	system("pause");
	cout << "\n\r" << endl;

	//practice_2_buildNeuralNetwork bottom

	//test top------------------------------
	/*//test1 top
	Mat<double> A;
	A.load("mat1.txt");

	A.reshape(A.n_rows*A.n_cols, 1);

	cout << A << endl;
	system("pause");
	//test1 bottom*/

	/*//test2 top
	Mat<double> A;
	A.load("mat1.txt");

	A = vectorise(A);

	cout << A << endl;
	system("pause");
	//test2 bottom*/

	/*//test3 top
	Mat<double> A;
	A.load("mat1.txt");

	A = vectorise(A);
	A.reshape(3, 3);

	cout << A << endl;
	system("pause");
	//test3 bottom*/

	/*//test3 top
	Mat<double> A;
	A.load("mat1.txt");

	A = vectorise(A);
	A.row(5);

	cout << A.row(5) << endl;
	system("pause");
	//test3 bottom*/


	/*//test4 top
	Mat<double> A;
	A.load("mat1.txt");

	A = sigmoid(A);

	cout << A << endl;
	system("pause");
	//test4 bottom*/

	/*//test5 top
	Mat<double> A;
	A.load("mat1.txt");



	cout << A(3)<<A(2,2) << endl;
	system("pause");
	//test5 bottom*/

	/*//test6 top
	Mat<double> A;
	A.load("mat1.txt");

	Mat<double> B;
	B = log(A);

	cout << B << endl;
	system("pause");
	//test6 bottom*/

	/*//test7 top
	Mat<double> A;
	A.load("mat1.txt");

	cout << sum(A) << endl;
	system("pause");
	//test7 bottom*/

	/*//test8 top
	Mat<double> A;
	A.load("mat1.txt");

	Mat<double> B;
	B.ones(1, 1);

	cout << B << endl;
	system("pause");
	//test8 bottom*/

	/*//test9 top
	Mat<double> A;
	A.load("matlab_data_1.txt");

	cout <<A.n_rows<<"   " <<A.n_cols<< endl;
	system("pause");
	//test9 bottom*/

	/*//test10 top
	Mat<double> A;
	A.load("mat1.txt");

	cout <<A.rows(1,2)<<endl;
	system("pause");
	//test10 bottom*/


	/*//test11 top
	Mat<double> A;
	A.load("mat1.txt");

	cout <<A.n_elem<<endl;
	system("pause");
	//test11 bottom*/


	/*//test12 top
	Mat<double> A;
	A.load("mat1.txt");
	vec a = vectorise(A);

	cout << a << endl;
	cout <<norm_karl(a)<<endl;
	system("pause");
	//test12 bottom*/

	/*//test13 top
	Mat<double> A;
	Mat<double> B;
	A.load("mat1.txt");
	B = A;
	cout <<(A==B) <<"\n\r"<< endl;
	cout << mean(A == B) << "\n\r" << endl;
	//test13 */
	//test bottom------------------------------
	return 0;
}


