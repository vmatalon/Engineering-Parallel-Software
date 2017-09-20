#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <sys/time.h>
#include <time.h>
#include "counters.cpp"

void opt_simd_sgemm(float *Y, float *A, float *B, int n);
void opt_scalar1_sgemm(float *Y, float *A, float *B, int n);
void opt_scalar0_sgemm(float *Y, float *A, float *B, int n);
void naive_sgemm(float *Y, float *A, float *B, int n);

int main(int argc, char *argv[])
{

	//Initialize arrays and their size
	int n = (1<<10);
	float* A = new float[n*n];
	float* B = new float[n*n];
	float* Y = new float[n*n];

 	//Initialize a time counter 
 	struct timeval tv;
 	gettimeofday(&tv,0);
 	double start_time = tv.tv_sec+1e-6*tv.tv_usec;
 	gettimeofday(&tv,0);
 	double elapsed_time = (tv.tv_sec+1e-6*tv.tv_usec)-start_time;

	gettimeofday(&tv,0);
	start_time = tv.tv_sec+1e-6*tv.tv_usec;

	//opt_simd_sgemm(Y,A,B,n);
	//opt_scalar1_sgemm(Y,A,B,n);
	//opt_scalar0_sgemm(Y,A,B,n);
	naive_sgemm(Y,A,B,n);
	
	gettimeofday(&tv,0);
	elapsed_time = (tv.tv_sec+1e-6*tv.tv_usec)-start_time;

	printf("Time = %f\n",elapsed_time);
  	delete [] A;
  	delete [] B;
  	delete [] Y;
}
