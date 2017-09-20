#include <pthread.h>
#include <cassert>
#include <math.h>

typedef struct
{
  double **a;
  double **b;
  double **c;
  int start;
  int end;
} worker_t;

void *matmuld_worker(void *arg)
{
  worker_t *t = static_cast<worker_t*>(arg);
  double **a = t->a;
  double **b = t->b;
  double **c = t->c;
  int start = t->start;
  int end = t->end;
  for(int i=start;i<end;i++)
  {
    for(int j=0;j<1024;j++)
    {
      for(int k=0;k<1024;k++)
	    {
	      c[i][j] += a[i][k]*b[k][j];
	    }
    }
  }
}

void pthread_matmuld(double **a,
		     double **b,
		     double **c,
		     int nthr)
{
  /* CS194: use pthreads to launch 
   * matrix multply worker threads.
   *
   * The structure and worker function
   * are good hints...
   */
  int step = floor(1024/nthr);
  pthread_t *thr = new pthread_t[nthr];
  worker_t *tInfo = new worker_t[nthr];

  /* Start up threads. */
  for(int i=0;i<nthr;i++)
  {
    tInfo[i].a = a;
    tInfo[i].b = b;
    tInfo[i].c = c;
    tInfo[i].start = i*step;
    if(i==nthr-1) {
      tInfo[i].end = 1024;
    } else {
      tInfo[i].end = (i+1)*step;
    }
    pthread_create(&thr[i],NULL,matmuld_worker,tInfo+i);
  }

  /*
  tInfo[0].a = a;
  tInfo[0].b = b;
  tInfo[0].c = c;
  tInfo[0].start = 0;
  tInfo[0].end = 1024;
  matmuld_worker((void*)tInfo);
  */

  /* Synchronize the completion of threads. */
  for(int i=0;i<nthr;i++)
  {
    pthread_join(thr[i],NULL);
  }

  delete [] thr;
  delete [] tInfo;
}
