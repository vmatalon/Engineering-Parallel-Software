__kernel void incr (__global float *Y, int n)
{
	/* Get current work-item index */
	int idx = get_global_id(0);

	/* Increment value of vector Y at index by 1
	 * unless you exceed the index space */
	if(idx < n)
	{
		Y[idx] = Y[idx] + 1.0f;
	}
}
