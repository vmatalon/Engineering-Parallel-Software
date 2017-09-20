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

	//Initialize a clock cycle counter
	initTicks(c);
	uint64_t start_cycles = getTicks(c);
	uint64_t elapsed_cycles = getTicks(c)-start_cycles;

	start_cycles = getTicks(c);

	opt_simd_sgemm(Y,A,B,n);
	//opt_scalar1_sgemm(Y,A,B,n);
	//opt_scalar0_sgemm(Y,A,B,n);
	//naive_sgemm(Y,A,B,n);
	
	elapsed_cycles = getTicks(c)-start_cycles;

	printf("Cycles = %lu\n",elapsed_cycles);
  	delete [] A;
  	delete [] B;
  	delete [] Y;
}
