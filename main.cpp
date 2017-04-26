/*--------------------------------version information top-------------------------------------------
Started at 2016
Created by Karlxin(410824290@qq.com)
Github:https://github.com/Karlxin/OK003.git
OpenKarlGeneralArtificialIntelligence
Version:dev_000

features:To be continued...

Notice:Dev version created without large experiment.The comments for every single line of the code will
be added soon.Karl has a whole theory to explain the control law and guide for optimization of gains.



postscript:
I am very grateful to Andrew Ng for the dreams of Artificial Intelligence and the open but valuable
course of machine learning in Coursera.

To make your life better.Making others' life better also.

If you wanted experiments videos and theory,please send emails to Karl with 410824290@qq.com.

Your time is valuable.No time for us to waste.Do our best to build the artificial intelligence

I hope some day we will meet each other with our dreams achieved.

Good Luck!

EL PSY CONGROO
--------------------------------version information bottom-------------------------------------------*/

#include "main.h"

//for fminlbfgs top
Mat<int32_t> layer_size_fminlbfgs;
//layer_size_fminlbfgs = { 400,133 };//the W1:133*400.b1:133
//layer_size_fminlbfgs = layer_size.t();
double sparsityParam_fminlbfgs = 0.01;// desired average activation of the hidden units.
double lambda_fminlbfgs = 1;// weight decay parameter
double beta_fminlbfgs = 3;//weight of sparsity penalty term
Mat<double> patches_fminlbfgs;
Mat<uint32_t> Theta_indicator_UFLDL;
//for fminlbfgs bottom



//transmitting data from matlab to armadillo top
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
//transmitting data from matlab to armadillo bottom


//main top
int main(int argc, char** argv)
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


	/*//practice_4_Symmetry_Neural_Networks top
	//neuron:layer1_400,layer2_133,layer3_33,layer4_10,layer5_33,layer6_133,layer7_400
	Mat<int32_t> layer_size;
	layer_size = { 400,133,33,10};
	layer_size = layer_size.t();
	Mat<double> X;//the photo,20*20 pixel 5000 photos,we uniform these into float 1:0,size:5000*400
	Mat<double> y;//the label, we are about to recognize the arabic digits from integer 9:0,size:5000*1
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
	//practice_4_Symmetry_Neural_Networks bottom*/

	/*//practice_5_UFLDL top
	//neuron:layer1_8*8=64,layer2_25,layer3_8*8=64
	Mat<int32_t> layer_size;
	//layer_size = { 64,25};//the W1:25*64.
	layer_size = { 64,1 };//using to check gradient
	layer_size = layer_size.t();
	double sparsityParam = 0.01;// desired average activation of the hidden units.
	//double lambda = 0.0001;// weight decay parameter
	double lambda = 1;// weight decay parameter

	double beta = 3;//weight of sparsity penalty term
	Mat<double> patches;
	patches.load("patches.txt");
	Mat<double> theta;

	theta = UFLDL_init_rand(layer_size(0), layer_size(1));

	pair<Mat<double>, Mat<double>> theta_fX_pair;

	UFLDL_checkNNGradients(theta, layer_size(0), layer_size(1), lambda, sparsityParam, beta, patches);
	system("pause");

	theta_fX_pair=UFLDL_fmincg(theta, layer_size(0), layer_size(1),
		lambda, sparsityParam, beta, patches);

	Mat<double> W1;

	Mat<double> b1;


	b1 = theta_fX_pair.first.rows(2 * layer_size(0)*layer_size(1), 2 * layer_size(0)*layer_size(1) + layer_size(1) - 1);
	cout << b1.t() << "\n\r" << endl;

	W1 = reshape(theta_fX_pair.first.rows(0, layer_size(0)*layer_size(1) -1), layer_size(1), layer_size(0));
	cout << W1.n_rows<<"\n\r"<<W1.n_cols << endl;
	system("pause");
	//practice_5_UFLDL bottom*/

	/*//practice_6_DeepLearning top
	//first we pretrain neurons
	//pretrain,use self-encoding to compress data,the size to be 400*133
	Mat<int32_t> layer_size;
	layer_size = { 400,133};//the W1:133*400.b1:133
	layer_size = layer_size.t();
	double sparsityParam = 0.01;// desired average activation of the hidden units.
	double lambda = 1;// weight decay parameter
	double beta = 3;//weight of sparsity penalty term
	Mat<double> patches;
	std::ifstream file("matlab_data_v2.txt");//to load the labeled sample
	patches = load_mat<double>(file, "X").t();

	Mat<double> theta;

	theta = UFLDL_init_rand(layer_size(0), layer_size(1));

	pair<Mat<double>, Mat<double>> theta_fX_pair;
	theta_fX_pair=UFLDL_fmincg(theta, layer_size(0), layer_size(1),
	lambda, sparsityParam, beta, patches);
	Mat<double> W1;
	Mat<double> b1;

	b1 = theta_fX_pair.first.rows(2 * layer_size(0)*layer_size(1), 2 * layer_size(0)*layer_size(1) + layer_size(1) - 1);
	cout << b1.t() << "\n\r" << endl;

	W1 = reshape(theta_fX_pair.first.rows(0, layer_size(0)*layer_size(1) -1), layer_size(1), layer_size(0));
	cout << W1.n_rows<<"\n\r"<<W1.n_cols << endl;
	system("pause");



	//then we use the compressed data as input
	//neuron:layer1_133,layer2_33,layer3_10
	layer_size = { 133,33,10};
	layer_size = layer_size.t();
	Mat<double> X;//the photo,20*20 pixel 5000 photos,we uniform these into float 1:0,size:5000*400
	Mat<double> y;//the label, we are about to recognize the arabic digits from integer 9:0,size:5000*1
	Mat<double> Theta;//Theta1_133*401,Theta2_33*134,Theta3_10*34,Theta4_33*11,Theta5_133*34,Theta6_400*134
	Mat<double> nn_params;//a temp for theta
	X = sigmoid((W1*patches + repmat(b1, 1, patches.n_cols))).t();//data compress to 133 from 400
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
		initial_Theta(i) = randInitializeWeights(layer_size(i), layer_size(i + 1));
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
	//practice_6_DeepLearning bottom*/

	/*//practice_7_DeepLearning_fminlbfgs top
	//first we pretrain neurons
	//pretrain,use self-encoding to compress data,the size to be 400*133

	//for fminlbfgs top
	Mat<int32_t> layer_size;
	//layer_size = { 400,133 };//the W1:133*400.b1:133
	//layer_size = layer_size.t();
	//double sparsityParam = 0.01;// desired average activation of the hidden units.
	double lambda = 1;// weight decay parameter
	//double beta = 3;//weight of sparsity penalty term
	Mat<double> patches;
	//for fminlbfgs bottom

	//for fminlbfgs top
	//Mat<int32_t> layer_size_fminlbfgs;
	layer_size_fminlbfgs = { 400,40 };//the W1:133*400.b1:133
	layer_size_fminlbfgs = layer_size_fminlbfgs.t();
	layer_size = layer_size_fminlbfgs;
	//double sparsityParam_fminlbfgs = 0.01;// desired average activation of the hidden units.
	//double lambda_fminlbfgs = 1;// weight decay parameter
	//double beta_fminlbfgs = 3;//weight of sparsity penalty term
	//Mat<double> patches_fminlbfgs;
	//for fminlbfgs bottom

	std::ifstream file("matlab_data_v2.txt");//to load the labeled sample
	patches = load_mat<double>(file, "X").t();
	patches_fminlbfgs = patches;

	Mat<double> theta;

	theta = UFLDL_init_rand(layer_size(0), layer_size(1));


	vec new_theta;
	new_theta = fminlbfgs(theta, theta.n_rows);

	Mat<double> W1;
	Mat<double> b1;

	b1 = new_theta.rows(2 * layer_size(0)*layer_size(1), 2 * layer_size(0)*layer_size(1) + layer_size(1) - 1);
	cout << b1.t() << "\n\r" << endl;

	W1 = reshape(new_theta.rows(0, layer_size(0)*layer_size(1) - 1), layer_size(1), layer_size(0));
	cout << W1.n_rows << "\n\r" << W1.n_cols << endl;
	system("pause");



	//then we use the compressed data as input
	//neuron:layer1_133,layer2_33,layer3_10
	layer_size = { 40,33,10 };
	layer_size = layer_size.t();
	Mat<double> X;//the photo,20*20 pixel 5000 photos,we uniform these into float 1:0,size:5000*400
	Mat<double> y;//the label, we are about to recognize the arabic digits from integer 9:0,size:5000*1
	Mat<double> Theta;//Theta1_133*401,Theta2_33*134,Theta3_10*34,Theta4_33*11,Theta5_133*34,Theta6_400*134
	Mat<double> nn_params;//a temp for theta
	X = sigmoid((W1*patches + repmat(b1, 1, patches.n_cols))).t();//data compress to 133 from 400
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
		initial_Theta(i) = randInitializeWeights(layer_size(i), layer_size(i + 1));
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
	//practice_7_DeepLearning_fminlbfgs bottom*/

	//practice_8_DeepLearning_fminlbfgs_n top
#define Theta_num(k) ((layer_size(k+1))*((layer_size(k)+1)))
	//first we pretrain neurons
	//pretrain,use self-encoding to compress data,the size to be 400*133

	//for fminlbfgs top
	Mat<int32_t> layer_size;
	//layer_size = { 400,133 };//the W1:133*400.b1:133
	//layer_size = layer_size.t();
	//double sparsityParam = 0.01;// desired average activation of the hidden units.
	double lambda = 1;// weight decay parameter
					  //double beta = 3;//weight of sparsity penalty term
	Mat<double> patches;
	//for fminlbfgs bottom

	//for fminlbfgs top
	//Mat<int32_t> layer_size_fminlbfgs;
	layer_size_fminlbfgs = { 400,10,400 };
	layer_size_fminlbfgs = layer_size_fminlbfgs.t();
	layer_size = layer_size_fminlbfgs;
	//double sparsityParam_fminlbfgs = 0.01;// desired average activation of the hidden units.
	//double lambda_fminlbfgs = 1;// weight decay parameter
	//double beta_fminlbfgs = 3;//weight of sparsity penalty term
	//Mat<double> patches_fminlbfgs;
	//for fminlbfgs bottom

	std::ifstream file("matlab_data_v2.txt");//to load the labeled sample
	patches = load_mat<double>(file, "X").t();
	patches_fminlbfgs = patches;

	Mat<double> theta;


	Theta_indicator_UFLDL.zeros(layer_size.n_rows);
	int32_t i_init_Theta_indicator_UFLDL;
	Theta_indicator_UFLDL(0) = 0;

	for (i_init_Theta_indicator_UFLDL = 1;
		i_init_Theta_indicator_UFLDL < layer_size.n_rows;
		i_init_Theta_indicator_UFLDL++)
	{
		Theta_indicator_UFLDL(i_init_Theta_indicator_UFLDL) =
			Theta_indicator_UFLDL(i_init_Theta_indicator_UFLDL - 1) +
			Theta_num(i_init_Theta_indicator_UFLDL - 1);
	}

	theta = UFLDL_init_rand_n(layer_size);


	vec new_theta;
	new_theta = fminlbfgs(theta, theta.n_rows);

	field<Mat<double>> W(layer_size.n_rows - 1, 1);
	field<Mat<double>> b(layer_size.n_rows - 1, 1);
	field<Mat<double>>theta_field(layer_size.n_rows - 1, 1);

	for (int i = 0; i < layer_size.n_rows - 1; i++)
	{
		theta_field(i) = (reshape(new_theta.rows(Theta_indicator_UFLDL(i), \
			Theta_indicator_UFLDL(i + 1) - 1), layer_size(i + 1), (layer_size(i) + 1)));
		W(i) = theta_field(i).cols(0, theta_field(i).n_cols - 2);
		b(i) = theta_field(i).col(theta_field(i).n_cols - 1);
	}
	W.save("W.mat");
	b.save("B.mat");
	system("pause");
	//practice_8_DeepLearning_fminlbfgs_n bottom

	/*//practice_9_DeepLearning_fminlbfgs_n_show top
#define Theta_num(k) ((layer_size(k+1))*((layer_size(k)+1)))
	//first we pretrain neurons
	//pretrain,use self-encoding to compress data,the size to be 400*133

	//for fminlbfgs top
	Mat<int32_t> layer_size;
	//layer_size = { 400,133 };//the W1:133*400.b1:133
	//layer_size = layer_size.t();
	//double sparsityParam = 0.01;// desired average activation of the hidden units.
	double lambda = 1;// weight decay parameter
					  //double beta = 3;//weight of sparsity penalty term
	Mat<double> patches;
	//for fminlbfgs bottom

	//for fminlbfgs top
	//Mat<int32_t> layer_size_fminlbfgs;
	layer_size_fminlbfgs = { 400,33,33,33,10,33,33,33,400 };
	layer_size_fminlbfgs = layer_size_fminlbfgs.t();
	layer_size = layer_size_fminlbfgs;
	//double sparsityParam_fminlbfgs = 0.01;// desired average activation of the hidden units.
	//double lambda_fminlbfgs = 1;// weight decay parameter
	//double beta_fminlbfgs = 3;//weight of sparsity penalty term
	//Mat<double> patches_fminlbfgs;
	//for fminlbfgs bottom

	std::ifstream file("matlab_data_v2.txt");//to load the labeled sample
	patches = load_mat<double>(file, "X").t();
	patches_fminlbfgs = patches;

	field<Mat<double>> W(layer_size.n_rows - 1, 1);
	field<Mat<double>> b(layer_size.n_rows - 1, 1);

	W.load("W.mat");
	b.load("b.mat");

	Mat<double> theta_temp;
	theta_temp = join_vert(vectorise(W(0)), vectorise(b(0)));

	for (uint32_t i = 1; i < layer_size.n_rows - 1; i++)
	{
		theta_temp = join_vert(join_vert(theta_temp, vectorise(W(i))), vectorise(b(i)));
	}


	Theta_indicator_UFLDL.zeros(layer_size.n_rows);
	int32_t i_init_Theta_indicator_UFLDL;
	Theta_indicator_UFLDL(0) = 0;

	for (i_init_Theta_indicator_UFLDL = 1;
		i_init_Theta_indicator_UFLDL < layer_size.n_rows;
		i_init_Theta_indicator_UFLDL++)
	{
		Theta_indicator_UFLDL(i_init_Theta_indicator_UFLDL) =
			Theta_indicator_UFLDL(i_init_Theta_indicator_UFLDL - 1) +
			Theta_num(i_init_Theta_indicator_UFLDL - 1);
	}

	UFLDL_get_Cost_Grad_n_show(theta_temp, layer_size_fminlbfgs,
		lambda_fminlbfgs, sparsityParam_fminlbfgs, beta_fminlbfgs, patches_fminlbfgs, 2,
		Theta_indicator_UFLDL);

	//practice_9_DeepLearning_fminlbfgs_n_show bottom*/


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
//main bottom

//I am Hououin Kyouma,a mad scientist,and the destroyer of this world's ruling structure.
//Failure is out of the question.

//Okay.I believe in you.

//little butterfly do not be afraid,the turbulence can save the world.
//OORGNO CYSPLE


