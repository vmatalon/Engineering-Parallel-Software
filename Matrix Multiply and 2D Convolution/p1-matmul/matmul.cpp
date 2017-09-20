#include <omp.h> 

void matmuld(double **a, double **b, double **c)
{
	int j,k;
	#pragma omp parallel for private(j,k) collapse(3)
		for(int i=0;i<1024;i++)
			for(j=0;j<1024;j++)
				for(k=0;k<1024;k++)
					c[i][j] += a[i][k]*b[k][j];
}
