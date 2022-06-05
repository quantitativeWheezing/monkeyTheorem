/* Write files to verify the uniformity of rng
 * Has a cpu analogue of matchFileMultiGPU from monkeys.cu
 */

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

// includes, project
#include "helper_cuda.h"
#include "config.h"
#include "structs.h"
#include "io.h"
#include "monkeys_kernels.cuh"
#include "test.h"

//----------------------------------------------------------------------------//
//! Write the mean and stddev over nRuns of rng 
//! @param  arrLen  how many numbers are generated and queried
//! @param  twSize  number of characters in alphabet used
//! @param  nRuns  how many groups of random numbers to be generated
//! @param  counts  frequency of each number generated
//!                 [rows] : number :: [cols] : # of occurences 
//! @param  outName  name of output file
//----------------------------------------------------------------------------//
static void writeStats(const size_t arrLen,
    const unsigned int twSize, 
    const unsigned int nRuns,
    const unsigned int *counts,
    const char *outName)
{

  double *mu, *sigma, sum;
  mu = (double *)malloc(twSize*sizeof(double));
  sigma = (double *)malloc(twSize*sizeof(double));

  // compute mean relative frequency and stddev
  for(unsigned int i = 0; i < twSize; i++) {

    sum = 0;
    for(unsigned int run = 0; run < nRuns; run++) {
      sum += 1.*counts[i+run*twSize]/arrLen;
    }
    mu[i] = sum/nRuns;
    sum = 0;
    for(unsigned int run = 0; run < nRuns; run++) {
      sum += pow(1.*counts[i+run*twSize]/arrLen-mu[i],2);
    }
    sigma[i] = pow(sum/nRuns,.5);
  }

  // write results to a file
  FILE *stats = fopen(outName,"w");
  for(unsigned int i = 0; i < twSize; i++) {
    fprintf(stats, "%2u %.5f %.5f\n", i, mu[i], sigma[i]);
  }
  fclose(stats);

  // free the land
  free(mu);
  free(sigma);

}

//----------------------------------------------------------------------------//
//! CPU: generate numbers to match a given target
//! @param  targUint  uint array to be matched
//! @param  twSize  number of characters in alphabet used
//! @param  targLen  length of target string
//! @param  cpuNumsGenerated  count how many numbers have been generated
//----------------------------------------------------------------------------//
static void cpuGuesser(const unsigned int *targUint,
    const unsigned int twSize,
    const unsigned int targLen,
    unsigned int &cpuNumsGenerated)
{

  unsigned int currGuess;
  bool matchSoFar, foundMatch = false;
  while (!foundMatch) {
    matchSoFar = true;
    for(unsigned int j = 0; j < targLen; j++) {
      currGuess = (unsigned int)((double)rand()/
          ((double)(RAND_MAX)+1)*twSize);
      matchSoFar = matchSoFar && (currGuess == targUint[j]);
      cpuNumsGenerated++;
      if (matchSoFar && (j == targLen-1)) {
        foundMatch = true;
      }
    }
  }
}

//----------------------------------------------------------------------------//
//! GPU: verify uniformity of rng
//! @param  arrLen  how many numbers are generated and queried
//! @param  twSize  number of characters in alphabet used
//! @param  nRuns  how many groups of random numbers to be generated
//! @param  seed  change seed for curand
//----------------------------------------------------------------------------//
void gpuTestDist(const size_t arrLen,
    const unsigned int twSize, 
    const unsigned int nRuns,
    const unsigned int seed)
{

  const dim3 block(N_THREADS), grid(arrLen/N_THREADS);

  // (host side) allocate memory
  unsigned int *counts, *h_nums;
  counts = (unsigned int *)malloc(twSize*nRuns*sizeof(unsigned int));
  h_nums = (unsigned int *)malloc(arrLen*sizeof(unsigned int));

  // (dev side) allocate memory
  unsigned int *d_nums; 
  curandState *randState;
  checkCudaErrors(cudaMalloc((void **)&d_nums, arrLen*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **)&randState, arrLen*sizeof(curandState)));

  // initialize curand state
  kernInitRand<<<grid, block>>>(randState, seed, arrLen);

  // generate random numbers nRuns times
  for(unsigned int run = 0; run < nRuns; run++) {
    kernGenOnly<<<grid, block>>>(randState, arrLen, twSize, d_nums);
    checkCudaErrors(cudaMemcpy(h_nums, d_nums, arrLen*sizeof(unsigned int), 
        cudaMemcpyDeviceToHost));

    // count frequencies for this run
    memset(&counts[run*twSize], 0, twSize*sizeof(unsigned int));
    for(unsigned int i = 0; i < arrLen; i++) {
      counts[h_nums[i]+run*twSize]++;
    }
  }

  // write stats to file
  char outName[64]; strcpy(outName, outDir); strcat(outName, "rng_gpu.txt");
  writeStats(arrLen, twSize, nRuns, counts, outName);

  // free the land
  free(counts);
  free(h_nums);
  cudaFree(d_nums);
  cudaFree(randState);
}

//----------------------------------------------------------------------------//
//! CPU: verify uniformity of rng
//! @param  arrLen  how many numbers are generated and queried
//! @param  twSize  number of characters in alphabet used
//! @param  nRuns  how many groups of random numbers to be generated
//----------------------------------------------------------------------------//
void cpuTestDist(const size_t arrLen,
    const unsigned int twSize, 
    const unsigned int nRuns)
{
  
  // (host side) allocate memory
  unsigned int uiRoll, *counts = (unsigned int *)malloc(
      twSize*nRuns*sizeof(unsigned int));
  memset(counts, 0, twSize*sizeof(unsigned int));

  // generate random numbers nRuns times
  for(unsigned int run = 0; run < nRuns; run++) {

    // count frequencies
    memset(&counts[run*twSize], 0,
        twSize*sizeof(unsigned int));
    for(unsigned int k = 0; k < arrLen; k++) {
      uiRoll = (unsigned int)((double)rand()/
          ((double)(RAND_MAX)+1)*twSize);
      counts[uiRoll+run*twSize]++;
    }
  }

  // write stats to file
  char outName[64]; strcpy(outName, outDir); strcat(outName, "rng_cpu.txt");
  writeStats(arrLen, twSize, nRuns, counts, outName);

  // free the land
  free(counts);
}


//----------------------------------------------------------------------------//
//! CPU: Generate numbers to match all words in a file
//! @param  fileName  file to be randomly generated
//! @param  arrLen  how many numbers are generated and queried
//! @param  targLen  length of target string
//! @param  alph  which alphabet to use, determines twSize
//----------------------------------------------------------------------------//
void matchFileCPU(const char *fileName,
    const size_t arrLen,
    const unsigned int targLen,
    const unsigned int alph)
{

  // initialize io struct and write header of output file
  struct ioPar io = ioInit("cpu_out_", fileName, inpDir, outDir, alph);
  ioHeader(io, false);

  // host arrays
  char *targStr = (char *)malloc((targLen+1)*sizeof(char));
  targStr[targLen] = '\0';
  unsigned int *targUint = (unsigned int *)malloc(
      (targLen+1)*sizeof(unsigned int));
  unsigned int numsSoFar;

  // timing and convergence
  clock_t begin, end;
  double elapsed;
  bool finishedRead = false;
  unsigned int currWord = 0;
  ioColTitles(io, false);
  begin = clock();

  while (!finishedRead) {

    memcpy(targStr, &io.fChars[currWord], sizeof(char)*targLen);
    memcpy(targUint, &io.fUints[currWord], sizeof(unsigned int)*targLen);
    currWord += targLen;

    // see if we're done 
    finishedRead = currWord + targLen >= io.numChars;
    if (finishedRead) break;

    // generate numbers until we get a match
    numsSoFar = 0;
    cpuGuesser(targUint, io.twSize, targLen, numsSoFar);
    ioWord(io, targStr, targLen, numsSoFar, 0, false);
  }

  // print footer of output file
  end = clock();
  elapsed = (double)(end-begin)/(CLOCKS_PER_SEC);
  ioFooter(io, elapsed);

  // free the land
  ioFree(io);
  free(targStr);
  free(targUint);
}
