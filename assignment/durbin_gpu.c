#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "durbin.h"

#ifndef NTHREADS
#define NTHREADS 4
#endif

#ifndef NTHREADS_GPU
#define NTHREADS_GPU 1024
#endif

/* Array initialization. */
static void init_array(int n,
                       DATA_TYPE POLYBENCH_1D(r, N, n))
{
  int i;
  #pragma omp parallel for num_threads(NTHREADS)
  for (i = 0; i < n; i++)
  {
    r[i] = (i + 1) / n / 4.0;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n,
                        DATA_TYPE POLYBENCH_1D(out, N, n))

{
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
                          DATA_TYPE POLYBENCH_1D(r, N, n),
                          DATA_TYPE POLYBENCH_1D(out, N, n))
{
  int i, k;
  DATA_TYPE sum, beta, alpha;
  DATA_TYPE y[2][N];
#pragma omp target enter data map(to: r[0:N]) map(alloc: out[0:N], y)
#pragma scop
#pragma omp target teams num_teams(_PB_N / NTHREADS_GPU)
  {
    y[0][0] = r[0];
    beta = 1;
    alpha = r[0];
    for (k = 1; k < _PB_N; k++)
    {
      beta = beta - alpha * alpha * beta;
      sum = r[k];
      #pragma omp distribute parallel for reduction(+:sum) num_threads(NTHREADS_GPU) dist_schedule(static, NTHREADS_GPU)
      for (i = 0; i <= k - 1; i++)
        sum += r[k - i - 1] * y[(k - 1) % 2][i];
      alpha = -sum * beta;
      #pragma omp distribute parallel for num_threads(NTHREADS_GPU) dist_schedule(static, NTHREADS_GPU)
      for (i = 0; i <= k - 1; i++)
        y[k % 2][i] = y[(k - 1) % 2][i] + alpha * y[(k - 1) % 2][k - i - 1];
      y[k % 2][k] = alpha;
    }
    #pragma omp distribute parallel for num_threads(NTHREADS_GPU) dist_schedule(static, NTHREADS_GPU)
    for (i = 0; i < _PB_N; i++)
      out[i] = y[(_PB_N - 1) % 2][i];
  }
  #pragma omp target exit data map(from: out[0:N]) map(release: r[0:N])
}

static long long int hash_array(int n,  DATA_TYPE POLYBENCH_1D(out, N, n))
{
    long long int hash = 13;
    int i;

    for (i = 0; i < n; i++)
    {
        hash = hash * 37 + out[i];
    }
    return hash;
}

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(out, DATA_TYPE, N, n);

  printf("----init_array-----\n");
  polybench_start_instruments;
  /* Initialize array(s). */
  init_array(n,
             POLYBENCH_ARRAY(r));
  polybench_stop_instruments;
  polybench_print_instruments;

  printf("----kernel durbin-----\n");
  polybench_start_instruments;

  /* Run kernel. */
  kernel_durbin(n,
                POLYBENCH_ARRAY(r),
                POLYBENCH_ARRAY(out));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(out)));
#ifdef PRINT_HASH
  long long int h2 = hash_array(n, POLYBENCH_ARRAY(out));
  printf("%lli\n", h2);
#endif
#ifdef PRINT_ARRAY
  print_array(n, POLYBENCH_ARRAY(out));
#endif

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(r);
  POLYBENCH_FREE_ARRAY(out);

  return 0;
}
