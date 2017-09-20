#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cassert>
#include <cmath>

#include "clhelp.h"

void sqr_sgemm(float *Y, float *A, float *B, int n);

int main(int argc, char *argv[])
{
  std::string matmul_kernel_str;
 
  std::string matmul_name_str = 
    std::string("matmul");
  std::string matmul_kernel_file = 
    std::string("matmul.cl");

  cl_vars_t cv; 
  cl_kernel matmul;
  
  readFile(matmul_kernel_file,
	   matmul_kernel_str);
  
  initialize_ocl(cv);
  
  compile_ocl_program(matmul, cv, matmul_kernel_str.c_str(),
		      matmul_name_str.c_str());
  
  /* Allocate arrays on the host
   * and fill with random data */
  float *h_A, *h_B, *h_Y, *h_YY;
  cl_mem g_A, g_B, g_Y;
  int n = (1<<10);
  h_A = new float[n*n];
  assert(h_A);
  h_B = new float[n*n];
  assert(h_B);
  h_Y = new float[n*n];
  assert(h_Y);
  h_YY = new float[n*n];
  assert(h_YY);
  bzero(h_Y, sizeof(float)*n*n);
  bzero(h_YY, sizeof(float)*n*n);
  
  for(int i = 0; i < (n*n); i++)
    {
      h_A[i] = (float)drand48();
      h_B[i] = (float)drand48();
    }

  cl_int err = CL_SUCCESS;
  /* CS194: Allocate Buffers on the GPU. */
  /* This function creates a buffer object */ 
  g_Y = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,sizeof(float)*n*n,
    NULL,&err); //output matrix Y
  CHK_ERR(err);
  g_A = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,sizeof(float)*n*n,
    NULL,&err); //output matrix Y
  CHK_ERR(err);
  g_B = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,sizeof(float)*n*n,
    NULL,&err); //output matrix Y
  CHK_ERR(err);
  
  /* CS194: Copy data from host CPU to GPU */
  /* This function enqueues commands to write to a buffer 
   * object from host CPU to the GPU */
  err = clEnqueueWriteBuffer(cv.commands,g_Y,true,0,sizeof(float)*n*n,
           h_Y,0,NULL,NULL); //output matrix Y
  CHK_ERR(err); //check for error 
  err = clEnqueueWriteBuffer(cv.commands,g_A,true,0,sizeof(float)*n*n,
           h_A,0,NULL,NULL); //input matrix A
  CHK_ERR(err); //check for error 
  err = clEnqueueWriteBuffer(cv.commands,g_B,true,0,sizeof(float)*n*n,
           h_B,0,NULL,NULL); //input matrix B
  CHK_ERR(err); //check for error 

  /* CS194: Create appropriately sized workgroups */
  size_t global_work_size[2] = {n,n}; //
  size_t local_work_size[2] = {1,512};
  
  /* CS194: Set kernel arguments */
  err = clSetKernelArg(matmul,0,sizeof(cl_mem),&g_Y); //output matrix Y
  CHK_ERR(err); //check for error 
  err = clSetKernelArg(matmul,1,sizeof(cl_mem),&g_A); //input matrix A
  CHK_ERR(err); //check for error 
  err = clSetKernelArg(matmul,2,sizeof(cl_mem),&g_B); //input matrix B
  CHK_ERR(err); //check for error 
  err = clSetKernelArg(matmul,3,sizeof(int),&n); //size of matrices
  CHK_ERR(err); //check for error

  /* Get time before computation */
  double t0 = timestamp();

  /* CS194: Launch matrix multiply kernel */
  /* This function enqueues a command to execute a kernel on the GPU */
  err = clEnqueueNDRangeKernel(cv.commands, //command queue
             matmul, //kernel 
             2, //work_dim
             NULL, //global_work_offset
             global_work_size, //global_work_size
             local_work_size, //local_work_size
             0, //num_events_in_wait_list
             NULL, //event_wait_list
             NULL //event
             );
  CHK_ERR(err); //check for error

  /* Waits until all previously queued OpenCL commands in cv.commands 
   * have completed. */
  err = clFinish(cv.commands);
  CHK_ERR(err);

  /* Get computation time */
  t0 = timestamp()-t0;

  /* Read result of GPU on host CPU */
  /* Enqueue commands to read from a buffer object to host memory. */
  err = clEnqueueReadBuffer(cv.commands, g_Y, true, 0, sizeof(float)*n*n,
			    h_Y, 0, NULL, NULL);
  CHK_ERR(err);
  err = clFinish(cv.commands);
  CHK_ERR(err);

  double t1 = timestamp();
  sqr_sgemm(h_YY, h_A, h_B, n);
  t1 = timestamp()-t1;

  /* Check answer */
  bool er = false;
  for(int i = 0; i < (n*n); i++)
  {
    double d = h_YY[i] - h_Y[i];
    d *= d;
    if(d > 0.0001)
    {
      printf("CPU and GPU results do not match!\n");
      printf("error at %d :(\n", i);
	    er = true;
      break;
    }
  }
  if(!er) {
    printf("CPU and GPU results match\n");
  }

  /* Shut down the OpenCL runtime */
  uninitialize_ocl(cv);
  
  delete [] h_A; 
  delete [] h_B; 
  delete [] h_Y;
  delete [] h_YY;

  clReleaseMemObject(g_A); 
  clReleaseMemObject(g_B); 
  clReleaseMemObject(g_Y);
  
  /* Print performance */
  double gpu_flops_s = (2.0 * pow((double)n, 3.0)) / t0;
  printf("GPU: %g gflops/sec\n", gpu_flops_s / (1e9));

  double cpu_flops_s = (2.0 * pow((double)n, 3.0)) / t1;
  printf("CPU: %g gflops/sec\n", cpu_flops_s / (1e9));
  return 0;
}
