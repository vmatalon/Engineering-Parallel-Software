#include <cstdio>
#include <cstring>
#include <cstdlib>
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

  	//Initialize a hardware counter
	hwCounter_t c;
	c.init = false;
	
	//Initialize an instruction counter
	initInsns(c);
	uint64_t start_instructions = getInsns(c);
	uint64_t executed_instructions = getInsns(c)-start_instructions;

	start_instructions = getInsns(c);

	opt_simd_sgemm(Y,A,B,n);
	//opt_scalar1_sgemm(Y,A,B,n);
	//opt_scalar0_sgemm(Y,A,B,n);
	//naive_sgemm(Y,A,B,n);
	
	executed_instructions = getInsns(c)-start_instructions;

	printf("Instructions = %lu\n",executed_instructions);
  	delete [] A;
  	delete [] B;
  	delete [] Y;
}
