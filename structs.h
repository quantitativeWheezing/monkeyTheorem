/* Structs to organize variables */

#ifndef _STRUCTS_H_
#define _STRUCTS_H_

// includes, system
#include <stdio.h>
//#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

//----------------------------------------------------------------------------//
//! Boolean analogues of int2 and int4
//----------------------------------------------------------------------------//
struct bool2 {bool x,y;}; 
struct bool4 {bool x,y,z,w;};

//----------------------------------------------------------------------------//
//! Contains parameters for matching target strings
//----------------------------------------------------------------------------//
struct ManyMonkeys
{

  //**************************************//
  // host side members
  //**************************************//

  //! count how many numbers were generated to match target
  unsigned int numsSoFar;

  //! true (false) if d_Amatch (d_Bmatch) should be overwritten
  bool writeA;

  //! current string 
  char *h_targStr;

  //! uint translation of current string
  unsigned int *h_targUint;

  //! (host) was contiguous match found in generated numbers?
  bool *h_foundMatch;


  //**************************************//
  // device side members
  //**************************************//
  
  //! curand state
  curandState *randState;

  //! uint translation of current string
  unsigned int *d_targUint;

  //! track matches: to be overwritten as needed 
  bool *d_charMatch;

  //! arrays to be overwritten when recursive searches are performed
  bool *d_Amatch, *d_Bmatch;

  //**************************************//
  // unified members
  //**************************************//

  //! (uni) was contiguous match found in generated numbers?
  bool *u_foundMatch;

};

//----------------------------------------------------------------------------//
//! Contains data for reading text files and simplifies output
//----------------------------------------------------------------------------//
struct ioPar
{

  //! file for output data
  FILE *fOut;

  //! array for characters
  char *fChars;

  //! array for uint translation of characters
  unsigned int *fUints;

  //! number of valid characters in input file
  size_t numChars;

};

#endif  // #ifndef _STRUCTS_H_
