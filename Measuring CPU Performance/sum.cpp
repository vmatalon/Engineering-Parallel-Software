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

	//Get current time in cycles
	uint64_t start = getTicks(c1);

	//Get the number of the user
	//int n = atoi(argv[1]);

	//Compute sum
	long long sum = 0;
	//for(long long i=0; i<n; i++)
	for(long long i=0; i<10000;i++)
		sum += i;

	//Get current time in cycles
	uint64_t elapsed = getTicks(c1)-start;

	//Print stuff 
	printf("Sum is %lld\n",sum);
	printf("Time elapsed in Ticks %lu\n",elapsed);
}
