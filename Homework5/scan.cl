__kernel void update(__global int *in,
         __global int *block,
         int n)
{
  size_t idx = get_global_id(0);
  size_t tid = get_local_id(0);
  size_t dim = get_local_size(0);
  size_t gid = get_group_id(0);

  if(idx < n && gid > 0)
    {
      in[idx] = in[idx] + block[gid-1];
    }
}

__kernel void scan(__global int *in, 
       __global int *out, 
       __global int *bout,
       __local int *buf,
       int n)
{
  size_t idx = get_global_id(0);
  size_t tid = get_local_id(0);
  size_t dim = get_local_size(0);
  size_t gid = get_group_id(0);

  /* CS194: Write this kernel! */
  //Load data into local memory
  if(idx < n) {
    buf[tid] = in[idx];
  } else {
    buf[tid] = 0; //identity for sum 
  }
  //Synchronize 
  barrier(CLK_LOCAL_MEM_FENCE);

  //Loop through all offsets 
  __local int temp[128];
  for(int offset = 1;
    offset < dim;
    offset <<= 1) {

    //Aggregate in a temp array because we cannot assure that the work 
    //groups will run in order
    if(tid >= offset) {
      temp[tid] = buf[tid-offset]+buf[tid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Coppy values to the buffer 
    if(tid >= offset) {
      buf[tid] = temp[tid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  //Assign the group value in bout to be the lase element in the group 
  if(tid == dim-1) {
    bout[gid] = buf[tid];
  }

  //Copy scanned values to out
  out[idx] = buf[tid];

}
