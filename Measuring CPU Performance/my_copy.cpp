#include <sys/time.h>
#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc,char *argv[])
{
	//Get the number of the user
	int n = atoi(argv[1]);
	int array[n];
	int copied[n];

	for(int i=0;i<n;i++) 
	{
		array[i] = i;
	}

	struct timeval tv;
	gettimeofday(&tv,0);
	double start = tv.tv_sec+1e-6*tv.tv_usec;
	for(int i=0;i<n;i++) 
	{
		copied[i] = array[i];
	}
	gettimeofday(&tv,0);
	double elapsed = tv.tv_sec+1e-6*tv.tv_usec-start;
	//printf("Current time = %f seconds.\n",elapsed);

	int megbts = sizeof(int)*n*8e-6;
	//printf("Megabits copied = %d.\n",megbts);

	printf("mbs/sec = %f\n",megbts/elapsed);
}
