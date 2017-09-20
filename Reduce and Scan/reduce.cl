__kernel void reduce(__global int *in, __global int *out, 
		     __local int *buf, int n)
{
	size_t tid = get_local_id(0);
	size_t gid = get_group_id(0);
	size_t dim = get_local_size(0);
	size_t idx = get_global_id(0);

	// Load data into local memory
	if (idx < n) {
	    buf[tid] = in[idx];
	} else {
	    buf[tid] = 0; //identity for sum 
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Perform reduction	
	for(int offset = 1;
		offset < dim;
    	offset <<= 1) {
	    if (tid >= offset) {
	    	buf[tid] += buf[tid + offset];
	    }
	    barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(tid == 0) {
    	out[gid] = buf[0];
   	}

}
