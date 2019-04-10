#include "pch.h"

void randomize( double * p, int n ) 
{
	for ( int i = 0; i < n; i++ ) {
		p[i] = ( double )rand() / ( RAND_MAX );
	}
}

NN * createNN( int n, int h, int o ) 
{
	srand(time(NULL));
	NN * nn = new NN;
	
    nn->n = new int[3];
	nn->n[0] = n;
	nn->n[1] = h;
	nn->n[2] = o;
	nn->l = 3;

	nn->w = new double ** [nn->l - 1];
    

	for ( int k = 0; k < nn->l - 1; k++ ) 
    {
		nn->w[k] = new double * [nn->n[k + 1]];
		for ( int j = 0; j < nn->n[k + 1]; j++ ) 
        {
			nn->w[k][j] = new double[nn->n[k]];			
			randomize( nn->w[k][j], nn->n[k]);
			// BIAS
			//nn->w[k][j] = new double[nn->n[k] + 1];			
			//randomize( nn->w[k][j], nn->n[k] + 1 );
		}		
	}

	nn->y = new double * [nn->l];
	for ( int k = 0; k < nn->l; k++ ) {
		nn->y[k] = new double[nn->n[k]];
		memset( nn->y[k], 0, sizeof( double ) * nn->n[k] );
	}

	nn->in = nn->y[0];
	nn->out = nn->y[nn->l - 1];

	nn->d = new double * [nn->l];
	for ( int k = 0; k < nn->l; k++ ) {
		nn->d[k] = new double[nn->n[k]];
		memset( nn->d[k], 0, sizeof( double ) * nn->n[k] );
	}

	return nn;
}

void releaseNN( NN *& nn ) 
{
	for ( int k = 0; k < nn->l - 1; k++ ) {
		for ( int j = 0; j < nn->n[k + 1]; j++ ) {
			delete [] nn->w[k][j];
		}
		delete [] nn->w[k];
	}
	delete [] nn->w;
		
	for ( int k = 0; k < nn->l; k++ ) {
		delete [] nn->y[k];
	}
	delete [] nn->y;
	
	for ( int k = 0; k < nn->l; k++ ) {
		delete [] nn->d[k];
		
	}
	delete [] nn->d;

	delete [] nn->n;

	delete nn;
	nn = NULL;
}

void feedforward(NN * nn)
{
	for (int k = 0; k < nn->l; k++)
	{
		if (k == 0)
		{
			for (int i = 0; i < nn->n[k]; i++)
			{
				double tmp = 1.0 / (1.0 + exp(-LAMBDA * nn->in[i]));
				nn->y[k][i] = nn->in[i];
			}
		}
		else
		{
			for (int i = 0; i < nn->n[k]; i++)
			{
				double weight = 0.0;
				for (int j = 0; j < nn->n[k - 1]; j++)
				{
					weight += nn->w[k - 1][i][j] *  nn->y[k-1][j];
				}
				double tmp = 1 / (1 + exp(-LAMBDA * weight));
				nn->y[k][i] = tmp;
			}
		}
	}

	for (int o = 0; o < nn->n[nn->l-1]; o++)
	{
		nn->out[o] =  nn->y[nn->l - 1][o];
	}
}

double backpropagation( NN * nn, double * t ) 
{
	for (int k = nn->l-1; k >= 0; k--)//layers
	{
		if (k == nn->l-1)//output layer
		{
			for (int i = 0; i < nn->n[k]; i++)//neurons
			{
				double to = t[i];
				double y = nn->y[k][i];
				nn->d[k][i] = (to - nn->y[k][i]) * LAMBDA *   nn->y[k][i] * (1 - nn->y[k][i]);
			}
		}
		else if ( k != 0) //hidden layer
		{
			for (int i = 0; i < nn->n[k]; i++)//neurons
			{
				double errorResult = 0.0;
				for (int j = 0; j < nn->n[k+1]; j++)//neurons from upper layer
				{
					errorResult += nn->d[k + 1][j] * nn->w[k][j][i];
				}

				nn->d[k][i] = errorResult * LAMBDA *   nn->y[k][i] * (1 - nn->y[k][i]);
			}
		}
	}

	for (int k = 0; k < nn->l-1; k++)//layers
	{
		for (int i = 0; i < nn->n[k+1]; i++)//upper layer
		{
			for (int j = 0; j < nn->n[k]; j++)//lower layer
			{
				double oldWeight = nn->w[k][i][j];
				double d = nn->d[k + 1][i];
				double y = nn->y[k][i];
				double newWeight = ETA * d * y;
				nn->w[k][i][j] = newWeight;
			}

		}
	}
	
	
	double error = 0.0;
	for (int n = 0; n < nn->n[nn->l - 1]; n++)
	{
		error += pow(t[n] - nn->y[nn->l - 1][n], 2);
	}
	error = error * (1.0 / 2.0);

	return error;
}

void setInput( NN * nn, double * in, bool verbose ) 
{
	memcpy( nn->in, in, sizeof( double ) * nn->n[0] );

	if ( verbose ) {
		printf( "input=(" );
		for ( int i = 0; i < nn->n[0]; i++ ) {
			printf( "%0.3f", nn->in[i] );
			if ( i < nn->n[0] - 1 ) {
				printf( ", " );
			}
		}
		printf( ")\n" );
	}
}

int getOutput( NN * nn, bool verbose ) 
{	
    double max = 0.0;
    int max_i = 0;
    if(verbose) printf( " output=" );
	for ( int i = 0; i < nn->n[nn->l - 1]; i++ ) 
    {
		if(verbose) printf( "%0.3f ", nn->out[i] );
        if(nn->out[i] > max) {
            max = nn->out[i];
            max_i = i;
        }
	}
	if(verbose) printf( " -> %d\n" , max_i);
    if(nn->out[0] > nn->out[1] && nn->out[0] - nn->out[1] < 0.1) return 2;
    return max_i;
}
