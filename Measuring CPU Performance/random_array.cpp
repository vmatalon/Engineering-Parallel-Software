#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <time.h>
#include "counters.h"
#include "counters.cpp"

int main(int argc, char *argv[])
{
	//Initialize a hardware counter
	hwCounter_t c1;
	c1.init = false;
	initTicks(c1);

	uint64_t start = getTicks(c1);
	uint64_t elapsed = getTicks(c1)-start;
	
	int n = 0;
	int increment = 1024*8;
	int index;
	for(int k=0;k<250;k++) 
	{
		n += increment;

		//Initialize array in order
		int array[n];
		for(int i=0;i<n;i++) 
		{
			array[i] = i;
		}

		//Seed the random number generator with the current time 
		srand(time(NULL));

		//Shuffle array randomly 
		for(int i=0;i<n;i++) 
		{
			int r = rand() % n;
			int temp = array[i];
			array[i] = array[r];
			array[r] = temp;
		}

		//Begin pointer chasing 
		index = 0;
		start = getTicks(c1);
		for(int step=0;step<1048576;step++) 
		{
			index = array[index];
		}
		elapsed = getTicks(c1)-start;
		elapsed = elapsed / 1048576;

		printf("%d,%lu,%d\n",n*4,elapsed,index);
	}
}
