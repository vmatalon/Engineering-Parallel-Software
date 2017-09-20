#include <sys/time.h>
#include <time.h>
#include <emmintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

//Naive implementation
void memcpy(int *dst, int *src, int n)
{
  for(int i=0;i<n;i++) 
    dst[i] = src[i];
}

//SIMD w/o cache
void simd_memcpy(void *dst, void *src, size_t nbytes)
{
  size_t i;

  size_t ilen = nbytes/sizeof(int);
  size_t ilen_sm = ilen - ilen%16;

  char *cdst=(char*)dst;
  char *csrc=(char*)src;

  int * idst=(int*)dst;
  int * isrc=(int*)src;

  __m128i l0,l1,l2,l3;

  _mm_prefetch((__m128i*)&isrc[0], _MM_HINT_NTA);
  _mm_prefetch((__m128i*)&isrc[4], _MM_HINT_NTA);
  _mm_prefetch((__m128i*)&isrc[8], _MM_HINT_NTA);
  _mm_prefetch((__m128i*)&isrc[12], _MM_HINT_NTA);
  
  for(i=0;i<ilen_sm;i+=16)
    {
      l0 =  _mm_load_si128((__m128i*)&isrc[i+0]);
      l1 =  _mm_load_si128((__m128i*)&isrc[i+4]);
      l2 =  _mm_load_si128((__m128i*)&isrc[i+8]);
      l3 =  _mm_load_si128((__m128i*)&isrc[i+12]);
    
      _mm_prefetch((__m128i*)&isrc[i+16], _MM_HINT_NTA);
      _mm_prefetch((__m128i*)&isrc[i+20], _MM_HINT_NTA);
      _mm_prefetch((__m128i*)&isrc[i+24], _MM_HINT_NTA);
      _mm_prefetch((__m128i*)&isrc[i+28], _MM_HINT_NTA);

      _mm_stream_si128((__m128i*)&idst[i+0],  l0);
      _mm_stream_si128((__m128i*)&idst[i+4],  l1);
      _mm_stream_si128((__m128i*)&idst[i+8],  l2);
      _mm_stream_si128((__m128i*)&idst[i+12], l3);

    }

  for(i=ilen_sm;i<ilen;i++)
    {
      idst[i] = isrc[i];
    }

  for(i=(4*ilen);i<nbytes;i++)
    {
      cdst[i] = csrc[i];
    }
}

//SIMD w/ cache
void simd_memcpy_cache(void *dst, void *src, size_t nbytes)
{
  size_t i;
  size_t sm = nbytes - nbytes%sizeof(int);
  size_t ilen = nbytes/sizeof(int);
  size_t ilen_sm = ilen - ilen%16;

  //printf("nbytes=%zu,ilen=%zu,ilen_sm=%zu\n",
  //nbytes,ilen,ilen_sm);


  char *cdst=(char*)dst;
  char *csrc=(char*)src;

  int * idst=(int*)dst;
  int * isrc=(int*)src;

  __m128i l0,l1,l2,l3;

  _mm_prefetch((__m128i*)&isrc[0], _MM_HINT_T0);
  _mm_prefetch((__m128i*)&isrc[4], _MM_HINT_T0);
  _mm_prefetch((__m128i*)&isrc[8], _MM_HINT_T0);
  _mm_prefetch((__m128i*)&isrc[12], _MM_HINT_T0);
  
  for(i=0;i<ilen_sm;i+=16)
    {
      l0 =  _mm_load_si128((__m128i*)&isrc[i+0]);
      l1 =  _mm_load_si128((__m128i*)&isrc[i+4]);
      l2 =  _mm_load_si128((__m128i*)&isrc[i+8]);
      l3 =  _mm_load_si128((__m128i*)&isrc[i+12]);
    
      _mm_prefetch((__m128i*)&isrc[i+16], _MM_HINT_T0);
      _mm_prefetch((__m128i*)&isrc[i+20], _MM_HINT_T0);
      _mm_prefetch((__m128i*)&isrc[i+24], _MM_HINT_T0);
      _mm_prefetch((__m128i*)&isrc[i+28], _MM_HINT_T0);

      _mm_store_si128((__m128i*)&idst[i+0],  l0);
      _mm_store_si128((__m128i*)&idst[i+4],  l1);
      _mm_store_si128((__m128i*)&idst[i+8],  l2);
      _mm_store_si128((__m128i*)&idst[i+12], l3);

    }

  for(i=ilen_sm;i<ilen;i++)
    {
      idst[i] = isrc[i];
    }

  for(i=(ilen*4);i<nbytes;i++)
    {
      cdst[i] = csrc[i];
    }
}

int main(int argc, char *argv[])
{
  //Initialize time counter 
  struct timeval tv;
  gettimeofday(&tv,0);
  double start = tv.tv_sec+1e-6*tv.tv_usec;
  gettimeofday(&tv,0);
  double elapsed = (tv.tv_sec+1e-6*tv.tv_usec)-start;

  int n = 0;
  int increment = 1024*8;
  double bytes;
  double megabits;
  for(int k=0;k<300;k++) 
  {
    n += increment;
    bytes = n*sizeof(int);
    megabits = bytes*8e-6;

    int* myarray = new int[n];
    int* copiedarray = new int[n];

    //Make myarray
    for(int i=0; i<n; i++) 
      myarray[i] = i;

    //Warm up cache
    //memcpy(copiedarray,myarray,n);
    //memcpy(copiedarray,myarray,n);
    //memcpy(copiedarray,myarray,n);
    //memcpy(copiedarray,myarray,n);
    //simd_memcpy(copiedarray,myarray,bytes);
    //simd_memcpy(copiedarray,myarray,bytes);
    //simd_memcpy(copiedarray,myarray,bytes);
    //simd_memcpy(copiedarray,myarray,bytes);
    simd_memcpy_cache(copiedarray,myarray,bytes);
    simd_memcpy_cache(copiedarray,myarray,bytes);
    simd_memcpy_cache(copiedarray,myarray,bytes);
    simd_memcpy_cache(copiedarray,myarray,bytes);

    //Get time
    gettimeofday(&tv,0);
    start = tv.tv_sec+1e-6*tv.tv_usec;

    //Copy
    //memcpy(copiedarray,myarray,n);
    //simd_memcpy(copiedarray,myarray,n*sizeof(int));
    simd_memcpy_cache(copiedarray,myarray,n*sizeof(int));
    
    //Get time
    gettimeofday(&tv,0);
    elapsed = (tv.tv_sec+1e-6*tv.tv_usec)-start;

    printf("%d,%f\n",n*4,megabits/elapsed);
    delete[] myarray;
    delete[] copiedarray;
  }
}
