#include <sys/time.h>
#include <time.h>
#include <cstdio>

int main(int argc,char *argv[])
{
	struct timeval tv;
	gettimeofday(&tv,0);
	double time = tv.tv_sec+1e-6*tv.tv_usec;
	printf("Current time = %f seconds.\n",time);
}
