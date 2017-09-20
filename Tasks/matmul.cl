__kernel void matmul(__global float *Y, __global float *A, __global float *B, 
	 int n)
{
	/* Get work item index */
	int index1 = get_global_id(0);
	int index2 = get_global_id(1);

	/* Multiply A and B and store in Y unless you exceed the index space */
	if((index1 < n) && (index2 < n))
	{
		int i = 0;
		while(i < n) {
			Y[index1*n+index2] += A[index1*n+i] * B[i*n+index2];
			i = i+1;
		}
		
	}
	
}
