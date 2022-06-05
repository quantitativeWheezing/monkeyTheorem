/* Functions for initializing structures, garbage collection, 
 * and running iterations
 */

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

// includes, project
#include "helper_cuda.h"
#include "config.h"
#include "structs.h"
#include "io.h"
#include "reduction_kernels.cuh"
#include "monkeys_kernels.cuh"
#include "monkeys.h"


//----------------------------------------------------------------------------//
//! Initializing function: make an instance of ManyMonkeys for each GPU
//! @param  monks  one instance for each GPU
//! @param  numGPUs  number of CUDA devices available
//! @param  allowedLetters  "lower" "upper" xor "all" letters to be used
//! @param  arrLen  how many numbers are generated and queried
//! @param  targLen  length of target string
//----------------------------------------------------------------------------//
static void initAllMonkeys(struct ManyMonkeys *monks,
    const int numGPUs,
    const size_t arrLen,
    const unsigned int targLen,
    const unsigned int seed,
    const unsigned int numThreads)
{

  const dim3 block(numThreads), grid(arrLen/numThreads);
  for(unsigned int i = 0; i < numGPUs; i++) {
    checkCudaErrors(cudaSetDevice(i));
    
    // (host side) allocate memory
    monks[i].h_targStr = (char *)malloc((targLen+1)*sizeof(char));
    monks[i].h_targStr[targLen] = '\0';
    monks[i].h_targUint = (unsigned int *)malloc(
        targLen*sizeof(unsigned int));
    monks[i].h_foundMatch = (bool *)malloc(sizeof(bool));
    
    // (dev side) allocate memory
    checkCudaErrors(cudaMalloc((void **)&monks[i].randState,
          arrLen*sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&monks[i].d_targUint,
          targLen*sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void **)&monks[i].d_charMatch,
          arrLen*sizeof(bool)));
    checkCudaErrors(cudaMalloc((void **)&monks[i].d_Amatch,
          arrLen*sizeof(bool)));
    checkCudaErrors(cudaMalloc((void **)&monks[i].d_Bmatch,
          arrLen*sizeof(bool)));
    
    // (unified) allocate memory
    checkCudaErrors(cudaMallocManaged((void **)&monks[i].u_foundMatch,
          sizeof(bool)));

    // initialize curand state
    kernInitRand<<<grid, block>>>(monks[i].randState, seed, arrLen);
  }
}

//----------------------------------------------------------------------------//
//! Frees memory from ManyMonkeys struct members (not the instance itself)
//! @param  monkeys  stores variables for calculation
//----------------------------------------------------------------------------//
static void freeMonkeysMem(struct ManyMonkeys &monkeys)
{

  // (host side) free memory
  free(monkeys.h_targStr);
  free(monkeys.h_targUint);
  free(monkeys.h_foundMatch);

  // (device side) free memory
  cudaFree(monkeys.randState);
  cudaFree(monkeys.d_targUint);
  cudaFree(monkeys.d_charMatch);
  cudaFree(monkeys.d_Amatch);
  cudaFree(monkeys.d_Bmatch);

  // (unified) free memory
  cudaFree(monkeys.u_foundMatch);
}


//----------------------------------------------------------------------------//
//! Use cpu to determine if contiguous matches were generated
//! @param  d_charMatch  rng array
//! @param  arrLen  how many numbers are generated and queried
//! @param  targLen  length of target string
//----------------------------------------------------------------------------//
static bool cpuCharMatch(const bool *d_charMatch,
    const size_t arrLen,
    const unsigned int targLen)
{

  bool *h_charMatch = (bool *)malloc(arrLen*sizeof(bool));
  checkCudaErrors(cudaMemcpy(h_charMatch, d_charMatch, arrLen*sizeof(bool), 
      cudaMemcpyDeviceToHost));
  bool matchSoFar;
  for(unsigned int offset = 0; offset < targLen; offset++) {
    for(unsigned int i = 0; i < arrLen/targLen; i++) {
      matchSoFar = true;
      for(unsigned int j = 0; j < targLen; j++) {
        unsigned int index = i*targLen+j+offset;
        if (index < arrLen){
          matchSoFar = matchSoFar && h_charMatch[index];
          if (matchSoFar && j == targLen-1) {
            free(h_charMatch); 
            return true;
          }
        }
      }
    }
  }

  free(h_charMatch);
  return false;
}

//----------------------------------------------------------------------------//
//! Generate numbers to match all words in a file
//! @param  fileName  file to be randomly generated
//! @param  numGPUs  number of CUDA devices available
//! @param  arrLen  how many numbers are generated and queried
//! @param  targLen  length of target string
//! @param  alph  which alphabet to use, determines twSize
//! @param  seed  change seed for curand
//! @param  numThreads  number of threads per block
//! @param  cpuCheck  if true, use the cpu to verify full matches
//----------------------------------------------------------------------------//
void matchFileMultiGPU(const char *fileName,
    const char *inpDir,
    const char *outDir,
    const int numGPUs,
    const size_t arrLen,
    const unsigned int targLen,
    const unsigned int alph,
    const unsigned int seed,
    const unsigned int numThreads,
    const bool cpuCheck)
{

  // initialize io struct and write header of output file
  struct ioPar io = ioInit("gpu_out_", fileName, inpDir, outDir, alph);
  ioHeader(io, true);

  // initialize structs
  struct ManyMonkeys *monks = (struct ManyMonkeys *)malloc(
      numGPUs*sizeof(struct ManyMonkeys));
  initAllMonkeys(monks, numGPUs, arrLen, targLen, seed, numThreads);

  // parameters to pass to kernels
  const dim3 block(numThreads), grid(arrLen/numThreads);
  const size_t totShared = (numThreads+1)*sizeof(unsigned int);
  cudaStream_t *stream = (cudaStream_t *)malloc(
      targLen*numGPUs*sizeof(cudaStream_t));

  // use this number to quickly identify contiguous matches
  long targetNum = 0;
  long fac = 1;
  for(unsigned int i = 0; i < targLen; i++) {targetNum += fac; fac *= 256;}

  // if requested, use cpu to verify results
  bool cpuRes[numGPUs];

  // assign streams to each gpu
  cudaDeviceProp deviceProp;
  for(unsigned int i = 0; i < numGPUs; i++) {
    checkCudaErrors(cudaSetDevice(i));
    for(unsigned int offset = 0; offset < targLen; offset++) {
      cudaStreamCreate(&stream[targLen*i+offset]);
    }
    cpuRes[i] = true;
    monks[i].h_foundMatch[0] = true;
    monks[i].u_foundMatch[0] = true;
    cudaGetDeviceProperties(&deviceProp, i);
    ioDevInfo(io, i, deviceProp.name);
  }

  // timing and convergence
  clock_t begin, end;
  double elapsed;
  bool finishedRead = false;
  unsigned int currWord = 0; 
  ioColTitles(io, true);
  begin = clock();

  while (!finishedRead) {

    // get next word 
    for(unsigned int i = 0; i < numGPUs; i++) {
      if (monks[i].h_foundMatch[0]) {
        checkCudaErrors(cudaSetDevice(i));

        memcpy(monks[i].h_targStr, &io.fChars[currWord],
            targLen*sizeof(char));
        memcpy(monks[i].h_targUint, &io.fUints[currWord],
            targLen*sizeof(unsigned int));
        currWord += targLen;

        // see if we're done 
        //finishedRead = currWord+targLen >= numChars;
        finishedRead = currWord+targLen >= io.numChars;
        if (finishedRead) break;

        // if we're not, copy word to device
        checkCudaErrors(cudaMemcpyAsync(monks[i].d_targUint,
            monks[i].h_targUint, targLen*sizeof(unsigned int), 
            cudaMemcpyHostToDevice, stream[targLen*i]));

        // host variable to see if match to word was found
        // we check the host variable often and use it to
        // minmize accesses to variable in unified mem
        monks[i].h_foundMatch[0] = false;
        monks[i].u_foundMatch[0] = false;

        // keep track of how many numbers have been generated
        monks[i].numsSoFar = 0;

        // alternate between overwriting Amatch and Bmatch
        monks[i].writeA = true;
        if (cpuCheck) cpuRes[i] = true;
      }
      if (finishedRead) break;
    }
    if (finishedRead) break;

    // generate numbers and test matches
    for(unsigned int i = 0; i < numGPUs; i++) {
      if(!monks[i].h_foundMatch[0]) {
        checkCudaErrors(cudaSetDevice(i));

        kernGenComp<<<grid, block, totShared, stream[targLen*i]>>>
          (monks[i].randState, arrLen,
           io.twSize, monks[i].d_targUint,
           targLen, monks[i].d_charMatch);
        cudaMemset(monks[i].d_Amatch, false, arrLen);
        cudaMemset(monks[i].d_Bmatch, false, arrLen);
        monks[i].numsSoFar += arrLen;

        if (cpuCheck) {
          cpuRes[i] = cpuCharMatch(monks[i].d_charMatch,
              arrLen, targLen);
        }
      }
    }

    // check contiguous matches
    // !!! we're being greedy here and trying to avoid a stream/device sync
    // !!! we launch stream[targLen*gpuID] last because the next step
    // !!! (reduction) is launched in that stream: hopefully it finishes last 
    for(unsigned int i = 0; i < numGPUs; i++) {
      unsigned int numBlocks = arrLen/(targLen*MATCH_VEC_SIZE*numThreads);
      if(!monks[i].h_foundMatch[0]) {
        checkCudaErrors(cudaSetDevice(i));
        for(unsigned int offset = 0; offset < targLen; offset++) {

#if TARGET_LENGTH == 4
          kernVec4Match<<<numBlocks, block, 0,
            stream[targLen*i+(targLen-1-offset)]>>>
            (monks[i].d_charMatch, arrLen, targLen,
             monks[i].d_Amatch+(arrLen*(targLen-1-offset))/targLen,
             targetNum, (targLen-1-offset)); 

#elif TARGET_LENGTH == 8
          kernVec8Match<<<numBlocks, block, 0,
            stream[targLen*i+(targLen-1-offset)]>>>
            (monks[i].d_charMatch, arrLen, targLen,
             monks[i].d_Amatch+(arrLen*(targLen-1-offset))/targLen,
             targetNum, (targLen-1-offset)); 

#endif
        }
        monks[i].writeA = false;
      }
    }

    // recursively search arrays for the presence of a contiguous match
    // note that kernels are launched in the inner loop:
    // we can achieve concurrency if neither GPU has found a match
    for (unsigned int s = arrLen; s > 1; s /= (numThreads*
          REDUCTION_VEC_SIZE)) {
      unsigned int numBlocks = (unsigned int)ceil((1.*s)/
          (numThreads*REDUCTION_VEC_SIZE));
      for(unsigned int i = 0; i < numGPUs; i++) {
        if(!monks[i].h_foundMatch[0]) {
          checkCudaErrors(cudaSetDevice(i));

          if (monks[i].writeA) {
            callVecAnyMatch(numBlocks, s, monks[i].d_Bmatch,
                monks[i].d_Amatch, totShared, stream[targLen*i],
                monks[i].u_foundMatch);
            monks[i].writeA = false;
          }

          else {
            callVecAnyMatch(numBlocks, s, monks[i].d_Amatch,
                monks[i].d_Bmatch, totShared, stream[targLen*i],
                monks[i].u_foundMatch);
            monks[i].writeA = true;
          }

          if (cpuCheck) checkCudaErrors(cudaDeviceSynchronize());
        }
      }
    }

    // was a contiguous match found in all the numbers generated?
    // if requested, use cpu to verify gpu results
    for(unsigned int i = 0; i < numGPUs; i++) {
      if(!monks[i].h_foundMatch[0]) {
        checkCudaErrors(cudaSetDevice(i));
        monks[i].h_foundMatch[0] = monks[i].u_foundMatch[0];
        
        if(monks[i].h_foundMatch[0]) {
          if (cpuCheck && cpuRes[i] != monks[i].h_foundMatch[0]) {
            printf("\nGPU produced false positive\n\n");
            exit(EXIT_FAILURE);
          }

          // if a match was found, write info to output file
          ioWord(io, monks[i].h_targStr, targLen, monks[i].numsSoFar, i, true);
        }

        else{
          if (cpuCheck && cpuRes[i] != monks[i].h_foundMatch[0]) {
            printf("\nGPU produced false negative\n\n");
            exit(EXIT_FAILURE);
          }
        }
      }
    }
  }

  // print footer of output file
  end = clock();
  elapsed = (double)(end-begin)/CLOCKS_PER_SEC;
  ioFooter(io, elapsed);

  // free the land
  for(unsigned int i = 0; i < numGPUs; i++) {
    checkCudaErrors(cudaSetDevice(i));
    freeMonkeysMem(monks[i]);
    checkCudaErrors(cudaDeviceSynchronize());
    for(unsigned int offset = 0; offset < targLen; offset++) {
      checkCudaErrors(cudaStreamSynchronize(stream[targLen*i+offset]));
      cudaStreamDestroy(stream[targLen*i+offset]);
    }
  }

  ioFree(io);
  free(stream);
  free(monks);
}
