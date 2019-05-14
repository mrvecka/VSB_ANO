#include "pch.h"
#include "NNClassification.h"

void train(NN* nn)
{
	int n =1000;
	double ** trainingSet = new double *[n];
	for (int i = 0; i < n; i++) {
		trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];

		bool classA = i % 2;

		for (int j = 0; j < nn->n[0]; j++) {
			if (classA) {
				trainingSet[i][j] = 0.1 * (double)rand() / (RAND_MAX)+0.6;
			}
			else {
				trainingSet[i][j] = 0.1 * (double)rand() / (RAND_MAX)+0.2;
			}
		}

		trainingSet[i][nn->n[0]] = (classA) ? 1.0 : 0.0;
		trainingSet[i][nn->n[0] + 1] = (classA) ? 0.0 : 1.0;
	}

	int i = 0;
	double error = 1.0;
	while (error > 0.01)
	{
		int k = i % n;
		setInput(nn, trainingSet[k]);
		feedforward(nn);
		backpropagation(nn, &trainingSet[i%n][nn->n[0]]);
		updateWeights(nn);
		error = computeError(nn, &trainingSet[i%n][nn->n[0]]);
		i++;
		printf("\rerr=%0.3f", error);

	}
	printf(" (%i iterations) result error: %f\n", i, error);

	for (int i = 1; i < n; i++) {
		delete[] trainingSet[i];
	}
	delete[] trainingSet;

}

void trainMyData(NN* nn, ObjectFeature feature)
{
	

	int n = feature.Objects.size();
	double ** trainingSet = new double *[n];

	int i = 0;
	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];
		for (int j = 0; j < nn->n[0]; j++)
		{
			if (j%nn->n[0] == 0)
				trainingSet[i][j] = (*obj).Feature1;
			if (j%nn->n[0] == 1)
				trainingSet[i][j] = (*obj).Feature2;
			//add more features
		}

		if ((*obj).ClassLabel.label == 1)//square
		{
			trainingSet[i][nn->n[0]] = 1.0;
			trainingSet[i][nn->n[0] + 1] = 0.0;
			trainingSet[i][nn->n[0] + 2] = 0.0;
		}
		else if ((*obj).ClassLabel.label == 2)//star
		{
			trainingSet[i][nn->n[0]] = 0.0;
			trainingSet[i][nn->n[0] + 1] = 1.0;
			trainingSet[i][nn->n[0] + 2] = 0.0;
		}
		else if ((*obj).ClassLabel.label == 3)//rectangle
		{
			trainingSet[i][nn->n[0]] = 0.0;
			trainingSet[i][nn->n[0] + 1] = 0.0;
			trainingSet[i][nn->n[0] + 2] = 1.0;
		}
		else
		{
			trainingSet[i][nn->n[0]] = 0.0;
			trainingSet[i][nn->n[0] + 1] = 0.0;
			trainingSet[i][nn->n[0] + 2] = 0.0;
		}

		i++;
		obj++;
	}
	
}


void test(NN* nn,int num_samples = 10)
{
	double* in = new double[nn->n[0]];
	int num_err = 0;
	for (int n = 0; n < num_samples; n++)
	{
		bool classA = rand() % 2;

		for (int j = 0; j < nn->n[0]; j++)
		{
			if (classA)
			{
				in[j] = 0.1 * (double)rand() / (RAND_MAX)+0.6;
			}
			else
			{
				in[j] = 0.1 * (double)rand() / (RAND_MAX)+0.2;
			}
		}
		printf("predicted: %d\n", !classA);
		setInput(nn, in, true);

		feedforward(nn);
		int output = getOutput(nn, true);
		if (output == classA) num_err++;
		printf("\n");
	}
}

void testMyData(NN* nn, ObjectFeature feature, int num_samples = 10)
{
	double* in = new double[nn->n[0]];

	

	int num_err = 0;

	int i = 0;
	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		for (int j = 0; j < nn->n[0]; j++)
		{
			if (j%nn->n[0] == 0)
				in[j] = (*obj).Feature1;
			if (j%nn->n[0] == 1)
				in[j] = (*obj).Feature2;
			//add more features
		}
		int classA = (*obj).ClassLabel.label;


		printf("predicted: %d\n", classA);
		setInput(nn, in, true);

		feedforward(nn);
		int output = getOutput(nn, true);
		if (output == classA) num_err++;
		printf("\n");
		i++;
		obj++;
	}


	double err = (double)num_err / num_samples;
	printf("test error: %.2f\n", err);
}

void train()
{
	NN * nn = createNN(2, 4, 2);
	train(nn);

	getchar();

	test(nn, 100);

	getchar();

	releaseNN(nn);

}

void trainFeatures(ObjectFeature feature)
{
	NN * nn = createNN(2, 4, 3);
	trainMyData(nn, feature);

	getchar();

	testMyData(nn, feature, 100);

	getchar();

	releaseNN(nn);

}
