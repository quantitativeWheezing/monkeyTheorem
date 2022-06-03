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
#include "parse_text.h"
#include "monkeys_kernels.cuh"
#include "test.h"

//----------------------------------------------------------------------------//
//! Write the mean and stddev over nRuns of rng 
//! @param  arrLen  how many numbers are generated and queried
//! @param  typewriterSize  number of characters in alphabet used
//! @param  nRuns  how many groups of random numbers to be generated
//! @param  counts  frequency of each number generated
//!                 [rows] : number :: [cols] : # of occurences 
//! @param  outName  name of output file
//----------------------------------------------------------------------------//
static void writeStats(const size_t arrLen,
    const unsigned int typewriterSize, 
    const unsigned int nRuns,
    const unsigned int *counts,
    const char *outName)
{

  double *mu, *sigma, sum;
  mu = (double *)malloc(typewriterSize*sizeof(double));
  sigma = (double *)malloc(typewriterSize*sizeof(double));

  // compute mean relative frequency and stddev
  for(unsigned int i = 0; i < typewriterSize; i++) {

    sum = 0;
    for(unsigned int run = 0; run < nRuns; run++) {
      sum += 1.*counts[i+run*typewriterSize]/arrLen;
    }
    mu[i] = sum/nRuns;
    sum = 0;
    for(unsigned int run = 0; run < nRuns; run++) {
      sum += pow(1.*counts[i+run*typewriterSize]/arrLen-mu[i],2);
    }
    sigma[i] = pow(sum/nRuns,.5);
  }

  // write results to a file
  FILE *stats = fopen(outName,"w");
  for(unsigned int i = 0; i < typewriterSize; i++) {
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
//! @param  typewriterSize  number of characters in alphabet used
//! @param  targetLen  length of target string
//! @param  cpuNumsGenerated  count how many numbers have been generated
//----------------------------------------------------------------------------//
static void cpuGuesser(const unsigned int *targUint,
    const unsigned int typewriterSize,
    const unsigned int targetLen,
    unsigned int &cpuNumsGenerated)
{
  unsigned int currGuess;
  bool matchSoFar, foundMatch = false;
  while (!foundMatch) {
    matchSoFar = true;
    for(unsigned int j = 0; j < targetLen; j++) {
      currGuess = (unsigned int)((double)rand()/
          ((double)(RAND_MAX)+1)*typewriterSize);
      matchSoFar = matchSoFar && (currGuess == targUint[j]);
      cpuNumsGenerated++;
      if (matchSoFar && (j == targetLen-1)) {
        foundMatch = true;
      }
    }
  }
}

//----------------------------------------------------------------------------//
//! GPU: verify uniformity of rng
//! @param  arrLen  how many numbers are generated and queried
//! @param  typewriterSize  number of characters in alphabet used
//! @param  nRuns  how many groups of random numbers to be generated
//! @param  seed  change seed for curand
//----------------------------------------------------------------------------//
void gpuTestDist(const size_t arrLen,
    const unsigned int typewriterSize, 
    const unsigned int nRuns,
    const unsigned int seed)
{

  const dim3 block(SHMEM_SIZE), grid(arrLen/SHMEM_SIZE);

  // (host side) allocate memory
  unsigned int *counts, *h_nums;
  counts = (unsigned int *)malloc(typewriterSize*nRuns*sizeof(unsigned int));
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
    kernGenOnly<<<grid, block>>>(randState, arrLen, typewriterSize, d_nums);
    checkCudaErrors(cudaMemcpy(h_nums, d_nums, arrLen*sizeof(unsigned int), 
        cudaMemcpyDeviceToHost));

    // count frequencies for this run
    memset(&counts[run*typewriterSize], 0, typewriterSize*sizeof(unsigned int));
    for(unsigned int i = 0; i < arrLen; i++) {
      counts[h_nums[i]+run*typewriterSize]++;
    }
  }

  // write stats to file
  char outName[64]; strcpy(outName, outDir); strcat(outName, "rng_gpu.txt");
  writeStats(arrLen, typewriterSize, nRuns, counts, outName);

  // free the land
  free(counts);
  free(h_nums);
  cudaFree(d_nums);
  cudaFree(randState);
}

//----------------------------------------------------------------------------//
//! CPU: verify uniformity of rng
//! @param  arrLen  how many numbers are generated and queried
//! @param  typewriterSize  number of characters in alphabet used
//! @param  nRuns  how many groups of random numbers to be generated
//----------------------------------------------------------------------------//
void cpuTestDist(const size_t arrLen,
    const unsigned int typewriterSize, 
    const unsigned int nRuns)
{
  
  // (host side) allocate memory
  unsigned int uiRoll, *counts = (unsigned int *)malloc(
      typewriterSize*nRuns*sizeof(unsigned int));
  memset(counts, 0, typewriterSize*sizeof(unsigned int));

  // generate random numbers nRuns times
  for(unsigned int run = 0; run < nRuns; run++) {

    // count frequencies
    memset(&counts[run*typewriterSize], 0,
        typewriterSize*sizeof(unsigned int));
    for(unsigned int k = 0; k < arrLen; k++) {
      uiRoll = (unsigned int)((double)rand()/
          ((double)(RAND_MAX)+1)*typewriterSize);
      counts[uiRoll+run*typewriterSize]++;
    }
  }

  // write stats to file
  char outName[64]; strcpy(outName, outDir); strcat(outName, "rng_cpu.txt");
  writeStats(arrLen, typewriterSize, nRuns, counts, outName);

  // free the land
  free(counts);
}


//----------------------------------------------------------------------------//
//! CPU: Generate numbers to match all words in a file
//! @param  fileName  file to be randomly generated
//! @param  arrLen  how many numbers are generated and queried
//! @param  typewriterSize  number of characters in alphabet used
//! @param  targetLen  length of target string
//----------------------------------------------------------------------------//
void matchFileCPU(const char *fileName,
    const size_t arrLen,
    const unsigned int targetLen,
    const unsigned int typewriterSize)
{

  // write header to output file
  char outName[64];
  strcpy(outName, outDir); strcat(outName, "cpu_out_");
  strcat(outName, fileName);
  FILE *fOut = fopen(outName, "w");
  time_t currentTime = time(NULL);
  fprintf(fOut, "%s", ctime(&currentTime));

  // translate file to uints and valid characters
  char inpName[64];
  strcpy(inpName, inpDir); strcat(inpName, fileName);
  const size_t numChars = getFileLen(inpName);
  unsigned int *fileUints = (unsigned int *)malloc(
      numChars*sizeof(unsigned int));
  char *fileChars = (char *)malloc((numChars+1)*sizeof(char));
  readFile(inpName, fileChars, fileUints);
  fprintf(fOut, "%s has %u valid characters\n", fileName,
      (unsigned int)numChars);
  fprintf(fOut, "Using a %2u character alphabet\n", typewriterSize);
  fprintf(fOut, "Translating %s with CPU\n\n", fileName);

  // host arrays
  char *h_targStr = (char *)malloc((targetLen+1)*sizeof(char));
  h_targStr[targetLen] = '\0';
  unsigned int *h_targUint = (unsigned int *)malloc(
      (targetLen+1)*sizeof(unsigned int));
  unsigned int totNumsGenerated;

  // timing and convergence
  clock_t begin, end;
  double elapsed;
  bool finishedRead = false;
  unsigned int currWord = 0;
  fprintf(fOut, "\nBeginning iterations\n\n");
  fprintf(fOut, "cpuOnly   TargetString     NumsGenerated\n");
  begin = clock();

  while (!finishedRead) {

    memcpy(h_targStr, &fileChars[currWord],
        sizeof(char)*targetLen);
    memcpy(h_targUint, &fileUints[currWord],
        sizeof(unsigned int)*targetLen);
    currWord += targetLen;

    // see if we're done 
    finishedRead = currWord + targetLen >= numChars;
    if (finishedRead) break;

    // generate numbers until we get a match
    totNumsGenerated = 0;
    cpuGuesser(h_targUint, typewriterSize, targetLen, totNumsGenerated);
    fprintf(fOut, "%s           %8s      %12u\n",
        "c", h_targStr, totNumsGenerated);

  }

  // print footer of output file
  end = clock();
  elapsed = (double)(end-begin)/(CLOCKS_PER_SEC);
  fprintf(fOut, "\nGenerating all text took %.3f seconds \n", elapsed);
  currentTime = time(NULL);
  fprintf(fOut, "%s", ctime(&currentTime));

  // free the land
  free(fileUints);
  free(fileChars);
  free(h_targStr);
  free(h_targUint);
  fclose(fOut);
}
