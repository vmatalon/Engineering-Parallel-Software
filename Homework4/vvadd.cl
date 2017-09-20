__kernel void vvadd (__global float *Y, __global float *A, __global float *B, 
	 int n)
{
	/* Get work item index */
	int index = get_global_id(0);

	/* Add the values of A and B at the specified index 
	 * and store in Y unless you exceed the index space */
	if(index < n)
	{
		Y[index] = A[index] + B[index];
	}
	
}
