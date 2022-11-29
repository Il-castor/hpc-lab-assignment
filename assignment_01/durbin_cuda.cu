#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <chrono>
#include <iostream>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "durbin.h"

#ifndef NTHREADS
#define NTHREADS 4
#endif
#define BLOCK_SIZE (32)

#include <cuda_runtime.h>
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d (code=%i, name=%s)\n", cudaGetErrorString(code), file, line, code, cudaGetErrorName(code));
        if (abort)
            exit(code);
    }
}

static void host_init_array(int n, DATA_TYPE *r)
{
  int i;
  #pragma omp parallel for
  for (i = 0; i < n; i++)
  {
    r[i] = (i + 1) / n / 4.0;
  }
}

/* Array initialization. */
__global__ static void init_array(int n, DATA_TYPE* r) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
    r[i] = (i + 1) / n / 4.0;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, DATA_TYPE* out) {
  int i;

  for (i = 0; i < n; i++)
  {
    fprintf(stderr, DATA_PRINTF_MODIFIER, out[i]);
    if (i % 20 == 0)
      fprintf(stderr, "\n");
  }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_durbin(int n,
                          DATA_TYPE *r,
                          DATA_TYPE *out)
{
  int i, k;
  DATA_TYPE sum, beta, alpha;
  DATA_TYPE y[2][N];
#pragma scop
  y[0][0] = r[0];
  beta = 1;
  alpha = r[0];


  for (k = 1; k < _PB_N; k++)
  {
    beta = beta - alpha * alpha * beta;

    sum = 0;

    // KERNEL  1
    for (i = 0; i <= k - 1; i++)
      sum += r[k - i - 1] * y[(k - 1) % 2][i];
    // END

    // reduce(sum)
    sum += r[k];
    alpha = -sum * beta;

    // KERNEL 2 (alpha)
    for (i = 0; i <= k - 1; i++)
      y[k % 2][i] = y[(k - 1) % 2][i] + alpha * y[(k - 1) % 2][k - i - 1];
    y[k % 2][k] = alpha;
    // END

    // barrier
  }

  for (i = 0; i < _PB_N; i++)
    out[i] = y[(_PB_N - 1) % 2][i];

}

__global__ static void durbin_k1(int k, DATA_TYPE *r, DATA_TYPE *y, DATA_TYPE  *tmp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k)
    tmp[i] = r[k - i - 1] * y[i];
}

/*
 * Applies a sum reduction on an entire warp, the result will be saved in sdata[0]
 * (the others will contain invalid data)
 */
__device__ void warpReduce(volatile DATA_TYPE *sdata, unsigned int tid) {
  if (BLOCK_SIZE >= 64) sdata[tid] += sdata[tid + 32];
  if (BLOCK_SIZE >= 32) sdata[tid] += sdata[tid + 16];
  if (BLOCK_SIZE >= 16) sdata[tid] += sdata[tid + 8];
  if (BLOCK_SIZE >= 8) sdata[tid] += sdata[tid + 4];
  if (BLOCK_SIZE >= 4) sdata[tid] += sdata[tid + 2];
  if (BLOCK_SIZE >= 2) sdata[tid] += sdata[tid + 1];
}

/*
 * every block applies a sum reduction and stores the result in g_odata
 */
__global__ static void blockReduce(DATA_TYPE *g_idata, DATA_TYPE *g_odata, unsigned int n) {
  __shared__ DATA_TYPE sdata[BLOCK_SIZE*2];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(BLOCK_SIZE*2) + tid;
  unsigned int gridSize = BLOCK_SIZE*2*gridDim.x;
  sdata[tid] = 0;
  while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+BLOCK_SIZE]; i += gridSize; }
  __syncthreads();
  if (BLOCK_SIZE >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (BLOCK_SIZE >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (BLOCK_SIZE >= 128) { if (tid < 64 ) { sdata[tid] += sdata[tid + 64];  } __syncthreads(); }
  if (tid < 32) warpReduce(sdata, tid);
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ static void durbin_k2(int k, DATA_TYPE *r, DATA_TYPE *yi, DATA_TYPE *yo, DATA_TYPE alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k)
    yo[i] = yi[i] + alpha * yi[k - i - 1];
  if (i == k)
    yo[k] = alpha;
}

static long long int hash_array(int n,  DATA_TYPE *out)
{
    long long int hash = 13;
    int i;

    for (i = 0; i < n; i++)
    {
        hash = hash * 37 + *reinterpret_cast<long long int *>(&out[i]);
    }
    return hash;
}

DATA_TYPE reduceCuda(DATA_TYPE *in, DATA_TYPE *tmp, int n) {
  // n -> ceil(n / BLOCK_SIZE) (repeat until we are 1)
  //printf("reduceCuda(%p, %p, %i)\n", (void *) in, (void *) tmp, n);

  while (n != 1) {
    int dimBlock = BLOCK_SIZE;
    int dimGrid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Divides "in" in block (by cuda), for every block, reduces it to only 1 number that writes in "out"
    blockReduce<<<dimGrid, dimBlock>>>(in, tmp, n);
    gpuErrchk(cudaPeekAtLastError());
    // Since every block writes only 1 element in "out", now we have "dimGrid" elements to reduce
    std::swap(in, tmp);
    n = dimGrid;
  }
  DATA_TYPE out = 0.0;
  gpuErrchk(cudaMemcpy(&out, in, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
  return out;
}


int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE *h_r = (DATA_TYPE *) malloc(sizeof(DATA_TYPE) * n);
  DATA_TYPE *h_out = (DATA_TYPE *) malloc(sizeof(DATA_TYPE) * n);
  DATA_TYPE *h_out2 = (DATA_TYPE *) malloc(sizeof(DATA_TYPE) * n);

  /* Initialize array(s). */
  auto begin = std::chrono::high_resolution_clock::now();
  host_init_array(n, h_r);

  /* Run kernel. */
  kernel_durbin(n, h_r, h_out);
  // code to benchmark

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << "s" << std::endl;

  /* Stop and print timer. */

  int dimBlock = BLOCK_SIZE;
  int dimGrid = (n + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE;

  DATA_TYPE *r, *out, *ycurr, *ynext, *red1, *red2;
  gpuErrchk(cudaMalloc((void **)&r, N * sizeof(DATA_TYPE)));
  gpuErrchk(cudaMalloc((void **)&out, N * sizeof(DATA_TYPE)));
  gpuErrchk(cudaMalloc((void **)&ycurr, N * sizeof(DATA_TYPE)));
  gpuErrchk(cudaMalloc((void **)&ynext, N * sizeof(DATA_TYPE)));
  gpuErrchk(cudaMalloc((void **)&red1, N * sizeof(DATA_TYPE)));
  gpuErrchk(cudaMalloc((void **)&red2, N * sizeof(DATA_TYPE)));


  auto begin2 = std::chrono::high_resolution_clock::now();

  init_array<<<dimGrid, dimBlock>>>(n, r);


  DATA_TYPE beta = 1;
  DATA_TYPE alpha = 0;
  gpuErrchk(cudaMemcpy(ycurr, r, sizeof(DATA_TYPE), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaMemcpy(&alpha, r, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));

  for (unsigned int k = 1; k < _PB_N; k++)
  {
    beta = beta - alpha * alpha * beta;

    durbin_k1<<<dimGrid, dimBlock>>>(k, r, ycurr, red1);
    gpuErrchk(cudaPeekAtLastError());
    DATA_TYPE sum = reduceCuda(red1, red2, k);

    DATA_TYPE rk;
    gpuErrchk(cudaMemcpy(&rk, &r[k], sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
    sum += rk;
    alpha = -sum * beta;

    durbin_k2<<<dimGrid, dimBlock>>>(k, r, ycurr, ynext, alpha);
    gpuErrchk(cudaPeekAtLastError());
    std::swap(ycurr, ynext);
  }

  gpuErrchk(cudaMemcpy(h_out2, ycurr, sizeof(DATA_TYPE) * n, cudaMemcpyDeviceToHost));


  auto end2 = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end2-begin2).count() << "s" << std::endl;

  gpuErrchk(cudaFree(r));
  gpuErrchk(cudaFree(out));
  gpuErrchk(cudaFree(ycurr));
  gpuErrchk(cudaFree(ynext));
  gpuErrchk(cudaFree(red1));
  gpuErrchk(cudaFree(red2));

  if (argc > 42 && ! strcmp(argv[0], "")) print_array(n, h_out2);
#ifdef PRINT_HASH
  long long int h1 = hash_array(n, h_out);
  long long int h2 = hash_array(n, h_out2);
  printf("%lli\n%lli\n", h1, h2);

  if (h1 != h2) {
    printf("\n\n\n-----------HASHS DIFFER-----------\n");
  }
  double diff = 0.0;
  for (int i = 0; i < n; i++) {
    double d = h_out[i] - h_out2[i];
    diff += d*d;
    std::cout << "INDEX " << i << " " <<  h_out[i] << " " <<  h_out2[i] << " diff:" << (h_out2[i] - h_out[i]) << std::endl;
  }
  std::cout << "NORM DIFFERENCE: " << diff << std::endl;
#endif
#ifdef PRINT_ARRAY
  print_array(n, h_out2);
#endif

  /* Be clean. */
  free(h_r);
  free(h_out);
  free(h_out2);

  return 0;
}
