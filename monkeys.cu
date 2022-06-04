/* Functions for initializing structures, garbage collection, 
 * and running iterations
 */

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

// includes, project
#include "helper_cuda.h"
#include "config.h"
#include "structs.h"
#include "parse_text.h"
#include "reduction_kernels.cuh"
#include "monkeys_kernels.cuh"
#include "monkeys.h"


//----------------------------------------------------------------------------//
//! Initializing function: make an instance of ManyMonkeys for each GPU
//! @param  monks  one instance for each GPU
//! @param  numGPUs  number of CUDA devices available
//! @param  allowedLetters  "lower" "upper" xor "all" letters to be used
//! @param  arrLen  how many numbers are generated and queried
//! @param  targetLen  length of target string
//----------------------------------------------------------------------------//
static void initAllMonkeys(struct ManyMonkeys *monks,
    const int numGPUs,
    const size_t arrLen,
    const unsigned int targetLen,
    const unsigned int seed)
{

  const dim3 block(SHMEM_SIZE), grid(arrLen/SHMEM_SIZE);

  for(unsigned int i = 0; i < numGPUs; i++) {
    checkCudaErrors(cudaSetDevice(i));
    
    // (host side) allocate memory
    monks[i].h_targStr = (char *)malloc((targetLen+1)*sizeof(char));
    monks[i].h_targStr[targetLen] = '\0';
    monks[i].h_targUint = (unsigned int *)malloc(
        targetLen*sizeof(unsigned int));
    monks[i].h_foundMatch = (bool *)malloc(sizeof(bool));
    
    // (dev side) allocate memory
    checkCudaErrors(cudaMalloc((void **)&monks[i].randState,
          arrLen*sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&monks[i].d_targUint,
          targetLen*sizeof(unsigned int)));
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
//! @param  targetLen  length of target string
//----------------------------------------------------------------------------//
static bool cpuCharMatch(const bool *d_charMatch,
    const size_t arrLen,
    const unsigned int targetLen)
{

  bool *h_charMatch = (bool *)malloc(arrLen*sizeof(bool));
  checkCudaErrors(cudaMemcpy(h_charMatch, d_charMatch, arrLen*sizeof(bool), 
      cudaMemcpyDeviceToHost));
  bool matchSoFar;
  for(unsigned int offset = 0; offset < targetLen; offset++) {
    for(unsigned int i = 0; i < arrLen/targetLen; i++) {
      matchSoFar = true;
      for(unsigned int j = 0; j < targetLen; j++) {
        unsigned int index = i*targetLen+j+offset;
        if (index < arrLen){
          matchSoFar = matchSoFar && h_charMatch[index];
          if (matchSoFar && j == targetLen-1) {
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
//! @param  targetLen  length of target string
//! @param  typewriterSize  number of characters in alphabet used
//! @param  seed  change seed for curand
//! @param  cpuCheck  if true, use the cpu to verify full matches
//----------------------------------------------------------------------------//
void matchFileMultiGPU(const char *fileName,
    const int numGPUs,
    const size_t arrLen,
    const unsigned int targetLen,
    const unsigned int typewriterSize,
    const unsigned int seed,
    const bool cpuCheck)
{

  // write header to output file
  char outName[64];
  strcpy(outName, outDir); strcat(outName, "gpu_out_");
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
  fprintf(fOut, "Translating %s with:\n", fileName);

  // initialize structs
  struct ManyMonkeys *monks = (struct ManyMonkeys *)malloc(
      numGPUs*sizeof(struct ManyMonkeys));
  initAllMonkeys(monks, numGPUs, arrLen, targetLen, seed);

  // parameters to pass to kernels
  const dim3 block(SHMEM_SIZE), grid(arrLen/SHMEM_SIZE);
  const size_t totShared = (SHMEM_SIZE+1)*sizeof(unsigned int);
  cudaStream_t *stream = (cudaStream_t *)malloc(targetLen*numGPUs*
      sizeof(cudaStream_t));

  // use this number to quickly identify contiguous matches
  long targetNum = 0;
  long fac = 1;
  for(unsigned int i = 0; i < targetLen; i++) {targetNum += fac; fac *= 256;}

  // if requested, use cpu to verify results
  bool cpuRes[numGPUs];

  // assign each gpu to a stream
  cudaDeviceProp deviceProp;
  for(unsigned int i = 0; i < numGPUs; i++) {
    checkCudaErrors(cudaSetDevice(i));
    for(unsigned int offset = 0; offset < targetLen; offset++) {
      cudaStreamCreate(&stream[targetLen*i+offset]);
    }
    cpuRes[i] = true;
    monks[i].h_foundMatch[0] = true;
    monks[i].u_foundMatch[0] = true;
    cudaGetDeviceProperties(&deviceProp, i);
    fprintf(fOut, "%u \"%s\"\n", i, deviceProp.name);
  }

  // timing and convergence
  clock_t begin, end;
  double elapsed;
  bool finishedRead = false;
  unsigned int currWord = 0; 
  fprintf(fOut, "\nBeginning iterations\n\n");
  fprintf(fOut, "gpuID     TargetString     NumsGenerated\n");
  begin = clock();

  while (!finishedRead) {

    // get next word 
    for(unsigned int i = 0; i < numGPUs; i++) {
      if (monks[i].h_foundMatch[0]) {
        checkCudaErrors(cudaSetDevice(i));

        memcpy(monks[i].h_targStr, &fileChars[currWord],
            targetLen*sizeof(char));
        memcpy(monks[i].h_targUint, &fileUints[currWord],
            targetLen*sizeof(unsigned int));
        currWord += targetLen;

        // see if we're done 
        finishedRead = currWord+targetLen >= numChars;
        if (finishedRead) break;

        // if we're not, copy word to device
        checkCudaErrors(cudaMemcpyAsync(monks[i].d_targUint,
            monks[i].h_targUint, targetLen*sizeof(unsigned int), 
            cudaMemcpyHostToDevice, stream[targetLen*i]));

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

        kernGenComp<<<grid, block, totShared, stream[targetLen*i]>>>
          (monks[i].randState, arrLen,
           typewriterSize, monks[i].d_targUint,
           targetLen, monks[i].d_charMatch);
        cudaMemset(monks[i].d_Amatch, false, arrLen);
        cudaMemset(monks[i].d_Bmatch, false, arrLen);
        monks[i].numsSoFar += arrLen;

        if (cpuCheck) {
          cpuRes[i] = cpuCharMatch(monks[i].d_charMatch,
              arrLen, targetLen);
        }
      }
    }

    // check contiguous matches
    // !!! we're being greedy here and trying to avoid a stream/device sync
    // !!! we launch stream[targetLen*gpuID] last because the next step
    // !!! (reduction) is launched in that stream: hopefully it finishes last 
    for(unsigned int i = 0; i < numGPUs; i++) {
      unsigned int numBlocks = arrLen/(targetLen*MATCH_VEC_SIZE*SHMEM_SIZE);
      if(!monks[i].h_foundMatch[0]) {
        checkCudaErrors(cudaSetDevice(i));
        for(unsigned int offset = 0; offset < targetLen; offset++) {

#if TARGET_LENGTH == 4
          kernVec4Match<<<numBlocks, block, 0,
            stream[targetLen*i+(targetLen-1-offset)]>>>
            (monks[i].d_charMatch, arrLen, targetLen,
             monks[i].d_Amatch+(arrLen*(targetLen-1-offset))/targetLen,
             targetNum, (targetLen-1-offset)); 

#elif TARGET_LENGTH == 8
          kernVec8Match<<<numBlocks, block, 0,
            stream[targetLen*i+(targetLen-1-offset)]>>>
            (monks[i].d_charMatch, arrLen, targetLen,
             monks[i].d_Amatch+(arrLen*(targetLen-1-offset))/targetLen,
             targetNum, (targetLen-1-offset)); 

#endif
        }
        monks[i].writeA = false;
      }
    }

    // recursively search arrays for the presence of a contiguous match
    // note that kernels are launched in the inner loop:
    // we can achieve concurrency if neither GPU has found a match
    //for (unsigned int s = arrLen/targetLen;  s > 1; s /= (SHMEM_SIZE*
    for (unsigned int s = arrLen; s > 1; s /= (SHMEM_SIZE*
          REDUCTION_VEC_SIZE)) {
      unsigned int numBlocks = (unsigned int)ceil((1.*s)/
          (SHMEM_SIZE*REDUCTION_VEC_SIZE));
      for(unsigned int i = 0; i < numGPUs; i++) {
        if(!monks[i].h_foundMatch[0]) {
          checkCudaErrors(cudaSetDevice(i));

          if (monks[i].writeA) {
            callVecAnyMatch(numBlocks, s, monks[i].d_Bmatch,
                monks[i].d_Amatch, totShared, stream[targetLen*i],
                monks[i].u_foundMatch);
            monks[i].writeA = false;
          }

          else {
            callVecAnyMatch(numBlocks, s, monks[i].d_Amatch,
                monks[i].d_Bmatch, totShared, stream[targetLen*i],
                monks[i].u_foundMatch);
            monks[i].writeA = true;
          }

          if (cpuCheck) checkCudaErrors(cudaDeviceSynchronize());
        }
      }
    }

    // was a contiguous match found in all the numbers generated?
    for(unsigned int i = 0; i < numGPUs; i++) {
      if(!monks[i].h_foundMatch[0]) {
        checkCudaErrors(cudaSetDevice(i));
        monks[i].h_foundMatch[0] = monks[i].u_foundMatch[0];
        
        if(monks[i].h_foundMatch[0]) {
          if (cpuCheck && cpuRes[i] != monks[i].h_foundMatch[0]) {
            printf("\nGPU produced false positive\n\n");
            exit(EXIT_FAILURE);
          }
          char *outString = removeSlash(monks[i].h_targStr, targetLen);
          fprintf(fOut, "%u           %8s      %12u\n", i,
              outString, monks[i].numsSoFar);
          free(outString); outString = NULL;
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
  fprintf(fOut, "\nGenerating all text took %.3f seconds \n", elapsed);
  currentTime = time(NULL);
  fprintf(fOut, "%s", ctime(&currentTime));

  // free the land
  free(fileUints);
  free(fileChars);
  for(unsigned int i = 0; i < numGPUs; i++) {
    checkCudaErrors(cudaSetDevice(i));
    freeMonkeysMem(monks[i]);
    checkCudaErrors(cudaDeviceSynchronize());
    for(unsigned int offset = 0; offset < targetLen; offset++) {
      checkCudaErrors(cudaStreamSynchronize(stream[targetLen*i+offset]));
      cudaStreamDestroy(stream[targetLen*i+offset]);
    }
  }
  free(stream);
  free(monks);
  fclose(fOut);
}
