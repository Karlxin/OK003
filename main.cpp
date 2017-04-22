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



	/*//practice_2_buildNeuralNetwork top
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

	Mat<double> pred;
	//bug inside predict
	pred = predict(Theta1, Theta2, X);

	//bug fix here.mean of a binary return also binary!!! so should change to double!!!
	cout << "\nTraining Set Accuracy:    \n " << (mean(conv_to<Mat<double>>::from(pred == y))) * 100 << endl;//bug here
	system("pause");
	cout << "\n\r" << endl;

	//practice_2_buildNeuralNetwork bottom*/



	/*//practice_3_buildMultipleNeuralNetwork top
	//neuron:input_400,middle1_133,middle2_33,output_10
	Mat<int32_t> layer_size;
	layer_size << 400 << endr << 133 << endr << 33 << endr << 10 << endr;

	Mat<double> X;//the photo,20*20 pixel 5000 photos,we uniform these into float 1:0,size:5000*400
	Mat<double> y;//the label, we are about to recognize the arabic digits from integer 9:0,size:5000*1
	Mat<double> Theta;//the wight,Theta1_133*401,Theta2_33*134,Theta3_10*34
	Mat<double> nn_params;
	double lambda;
	std::ifstream file("matlab_data_v2.txt");

	X = load_mat<double>(file, "X");
	y = load_mat<double>(file, "y");

	pair<double,Mat<double>>J_grad_pair_3;

	Mat<uint32_t> Theta_indicator;
	Theta_indicator.zeros(layer_size.n_rows);

	int32_t i_init_Theta_indicator;

#define Theta_num(k) ((layer_size(k+1))*((layer_size(k)+1)))

	Theta_indicator(0) = 0;

	for (i_init_Theta_indicator = 1; i_init_Theta_indicator < layer_size.n_rows; i_init_Theta_indicator++)
	{
		Theta_indicator(i_init_Theta_indicator) = Theta_indicator(i_init_Theta_indicator - 1) +
			Theta_num(i_init_Theta_indicator-1);
	}

	Mat<double> initial_Theta1;
	Mat<double> initial_Theta2;
	Mat<double> initial_Theta3;
	Mat<double> initial_nn_params;
	initial_Theta1 = randInitializeWeights(layer_size(0), layer_size(1));
	initial_Theta2 = randInitializeWeights(layer_size(1), layer_size(2));
	initial_Theta3 = randInitializeWeights(layer_size(2), layer_size(3));

	initial_nn_params = join_vert(vectorise(initial_Theta1), vectorise(initial_Theta2));
	initial_nn_params = join_vert(initial_nn_params, vectorise(initial_Theta3));

	lambda = 1;

	pair<Mat<double>, Mat<double>> fmincg_temp;

	fmincg_temp = fmincg_3(initial_nn_params, layer_size, X, y, lambda, Theta_indicator);

	Mat<double> pred;

	pred = predict_3(fmincg_temp.first, X, Theta_indicator,layer_size);

	cout << "\nTraining Set Accuracy:    \n " << (mean(conv_to<Mat<double>>::from(pred == y))) * 100 << endl;//bug here
	system("pause");
	cout << "\n\r" << endl;
	//practice_3_buildMultipleNeuralNetwork bottom*/

	
	//practice_4_Symmetry_Neural_Networks top
	//neuron:layer1_400,layer2_133,layer3_33,layer4_10,layer5_33,layer6_133,layer7_400
	Mat<int32_t> layer_size;
	layer_size = { 400,133,33,10};
	layer_size = layer_size.t();
	Mat<double> X;//the photo,20*20 pixel 5000 photos,we uniform these into float 1:0,size:5000*400
	Mat<double> y;//the label, we are about to recognize the arabic digits from integer 9:0,size:5000*1
	Mat<double> y_2;//the label, we are about to sparse self-coding,so y_2 labeled with 400 number,size:5000*1
	Mat<double> Theta;//Theta1_133*401,Theta2_33*134,Theta3_10*34,Theta4_33*11,Theta5_133*34,Theta6_400*134
	Mat<double> nn_params;//a temp for theta
	double lambda;//generalization term
	std::ifstream file("matlab_data_v2.txt");//to load the labeled sample
	X = load_mat<double>(file, "X");
	y = load_mat<double>(file, "y");
	pair<double, Mat<double>>J_grad_pair_n;
	Mat<uint32_t> Theta_indicator;
	Theta_indicator.zeros(layer_size.n_rows);
	int32_t i_init_Theta_indicator;

#define Theta_num(k) ((layer_size(k+1))*((layer_size(k)+1)))

	Theta_indicator(0) = 0;

	for (i_init_Theta_indicator = 1; i_init_Theta_indicator < layer_size.n_rows; i_init_Theta_indicator++)
	{
		Theta_indicator(i_init_Theta_indicator) = Theta_indicator(i_init_Theta_indicator - 1) +
			Theta_num(i_init_Theta_indicator - 1);
	}
	
	field<Mat<double>>initial_Theta(layer_size.n_rows - 1, 1);
	Mat<double> initial_nn_params;
	for (int i = 0; i < layer_size.n_rows - 1; i++)
	{
		initial_Theta(i)= randInitializeWeights(layer_size(i), layer_size(i+1));
	}

	initial_nn_params = join_vert(vectorise(initial_Theta(0)), vectorise(initial_Theta(1)));
	for (int i = 2; i < layer_size.n_rows - 1; i++)
	{
		initial_nn_params = join_vert(initial_nn_params, vectorise(initial_Theta(i)));
	}

	lambda = 1;
	pair<Mat<double>, Mat<double>> fmincg_temp;

	fmincg_temp = fmincg_n(initial_nn_params, layer_size, X, y, lambda, Theta_indicator);

	Mat<double> pred;

	pred = predict_n(fmincg_temp.first, X, Theta_indicator, layer_size);

	cout << "\nTraining Set Accuracy:    \n " << (mean(conv_to<Mat<double>>::from(pred == y))) * 100 << endl;//bug here
	system("pause");
	cout << "\n\r" << endl;
	//practice_4_Symmetry_Neural_Networks bottom


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
		//test13 bottom*/

		/*//test14 top
		Mat<double> A;
		Mat<double> B;
		Col<double> C;
		C << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 8 << 9;
		A.load("mat1.txt");
		B = reshape(C, 3, 3);
		cout <<B <<"\n\r"<< endl;
		//test14 bottom*/
		//test bottom------------------------------
	return 0;
}


