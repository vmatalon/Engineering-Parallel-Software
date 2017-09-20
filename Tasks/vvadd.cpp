#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "clhelp.h"

int main(int argc, char *argv[])
{
  std::string vvadd_kernel_str;

  /* Provide names of the OpenCL kernels
   * and cl file that they're kept in */
  std::string vvadd_name_str = 
    std::string("vvadd");
  std::string vvadd_kernel_file = 
    std::string("vvadd.cl");

  cl_vars_t cv; 
  cl_kernel vvadd;

  /* Read OpenCL file into STL string */
  readFile(vvadd_kernel_file,
	   vvadd_kernel_str);
  
  /* Initialize the OpenCL runtime 
   * Source in clhelp.cpp */
  initialize_ocl(cv);
  
  /* Compile all OpenCL kernels */
  compile_ocl_program(vvadd, cv, vvadd_kernel_str.c_str(),
		      vvadd_name_str.c_str());
  
  /* Arrays on the host (CPU) */
  float *h_A, *h_B, *h_Y;
  /* Arrays on the device (GPU) */
  cl_mem g_A, g_B, g_Y;

  /* Allocate arrays on the host
   * and fill with random data */
  int n = (1<<20);
  h_A = new float[n];
  h_B = new float[n];
  h_Y = new float[n];
  bzero(h_Y, sizeof(float)*n);
  
  for(int i = 0; i < n; i++)
    {
      h_A[i] = (float)drand48();
      h_B[i] = (float)drand48();
    }

  cl_int err = CL_SUCCESS;
  /* CS194: Allocate memory for arrays on the GPU */
  /* This function creates a buffer object */ 
  g_Y = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,sizeof(float)*n,NULL,&err); //output vector Y
  CHK_ERR(err); //check for error 
  g_A = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,sizeof(float)*n,NULL,&err); //input vector A
  CHK_ERR(err); //check for error 
  g_B = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,sizeof(float)*n,NULL,&err); //input vector B
  CHK_ERR(err); //check for error 

  /* CS194: Copy data from host CPU to GPU */
  /* This function enqueues commands to write to a buffer 
   * object from host CPU to the GPU */
  err = clEnqueueWriteBuffer(cv.commands,g_Y,true,0,sizeof(float)*n,
           h_Y,0,NULL,NULL); //output vector Y
  CHK_ERR(err); //check for error 
  err = clEnqueueWriteBuffer(cv.commands,g_A,true,0,sizeof(float)*n,
           h_A,0,NULL,NULL); //input vector A
  CHK_ERR(err); //check for error 
  err = clEnqueueWriteBuffer(cv.commands,g_B,true,0,sizeof(float)*n,
           h_B,0,NULL,NULL); //input vector B
  CHK_ERR(err); //check for error 
 
  /* CS194: Define the global and local workgroup sizes */
  size_t global_work_size[1] = {n}; //size of vectors
  size_t local_work_size[1] = {128}; //workgroup size 
  
  /* CS194: Set kernel arguments */
  err = clSetKernelArg(vvadd,0,sizeof(cl_mem),&g_Y); //output vector Y
  CHK_ERR(err); //check for error 
  err = clSetKernelArg(vvadd,1,sizeof(cl_mem),&g_A); //input vector A
  CHK_ERR(err); //check for error 
  err = clSetKernelArg(vvadd,2,sizeof(cl_mem),&g_B); //input vector B
  CHK_ERR(err); //check for error 
  err = clSetKernelArg(vvadd,3,sizeof(int),&n); //size of vectors
  CHK_ERR(err); //check for error 

  /* CS194: Call kernel on the GPU */
  /* This function enqueues a command to execute a kernel on the GPU */
  err = clEnqueueNDRangeKernel(cv.commands, //command queue
             vvadd, //kernel 
             1, //work_dim
             NULL, //global_work_offset
             global_work_size, //global_work_size
             local_work_size, //local_work_size
             0, //num_events_in_wait_list
             NULL, //event_wait_list
             NULL //event
             );
  CHK_ERR(err); //check for error 

  /* Read result of GPU on host CPU */
  /* Enqueue commands to read from a buffer object to host memory. */
  err = clEnqueueReadBuffer(cv.commands, g_Y, true, 0, sizeof(float)*n,
			    h_Y, 0, NULL, NULL);
  CHK_ERR(err); //check for error 

  /* Check answer */
  bool er = false;
  for(int i = 0; i < n; i++)
  {
    float d = h_A[i] + h_B[i];
    if(h_Y[i] != d)
    {
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
  
  clReleaseMemObject(g_A); 
  clReleaseMemObject(g_B); 
  clReleaseMemObject(g_Y);
  
  return 0;
}
