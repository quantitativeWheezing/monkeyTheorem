/* Helper function for making sure that filenames are valid
 * We also use a hashmap to map characters to numbers
 * Lastly, we define a function to replace '\n' with '\\' for output
 */

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <cassert>

// includes, project
#include "config.h"
#include "structs.h"
#include "io.h"

using std::unordered_map;

//----------------------------------------------------------------------------//
//! Check that pointer fileName is not null and is for a valid file
//! @param  fileName  pointer for filename to be read
//----------------------------------------------------------------------------//
static void checkFileName(const char *fileName)
{

  if (!fileName) {
    printf("fileName is null pointer\n"); exit(EXIT_FAILURE);
  }
  else {
    FILE *f = fopen(fileName, "r");
    if (!f) {printf("File %s not found\n", fileName); exit(EXIT_FAILURE);}
    else {fclose(f);}
  }
}

//----------------------------------------------------------------------------//
//! Hashmap to quickly convert strings to uint
//----------------------------------------------------------------------------//
static std::unordered_map<char, unsigned int> charMapInit(unsigned int alph)
{

  assert((alph == 0 || alph == 1 || alph == 2) && "alph must be in {0,1,2}");
  std::unordered_map<char, unsigned int> charMap
  {
    {'a', 0},{'b', 1},{'c', 2},{'d', 3},{'e', 4},{'f', 5},{'g', 6},{'h', 7},
    {'i', 8},{'j', 9},{'k',10},{'l',11},{'m',12},{'n',13},{'o',14},{'p',15},
    {'q',16},{'r',17},{'s',18},{'t',19},{'u',20},{'v',21},{'w',22},{'x',23},
    {'y',24},{'z',25} 
  };

  if (alph >= 1) {
    charMap['A']=26; charMap['B']=27; charMap['C']=28; charMap['D']=29; 
    charMap['I']=34; charMap['J']=35; charMap['K']=36; charMap['L']=37; 
    charMap['Q']=42; charMap['R']=43; charMap['S']=44; charMap['T']=45; 
    charMap['E']=30; charMap['F']=31; charMap['G']=32; charMap['H']=33; 
    charMap['M']=38; charMap['N']=39; charMap['O']=40; charMap['P']=41; 
    charMap['U']=46; charMap['V']=47; charMap['W']=48; charMap['X']=49; 
    charMap['Y']=50; charMap['Z']=51;
  }

  if (alph == 2) {
    charMap['0']=52; charMap['1']=53; charMap['2']=54; charMap['3']=55;
    charMap['4']=56; charMap['5']=57; charMap['6']=58; charMap['7']=59; 
    charMap['8']=60; charMap['9']=61; 
    charMap[' ']=62; charMap[',']=63; charMap[';']=64; charMap[':']=65;
    charMap['$']=70; charMap['%']=71; charMap['^']=72; charMap['&']=73;
    charMap['/']=78; charMap['=']=79; charMap['|']=80; charMap['(']=81;
    charMap['.']=66; charMap['!']=67; charMap['?']=68; charMap['#']=69; 
    charMap['_']=74; charMap['+']=75; charMap['-']=76; charMap['*']=77; 
    charMap[')']=82; charMap['[']=83; charMap[']']=84; charMap['{']=85; 
    charMap['}']=86; charMap['<']=87; charMap['>']=88; 
    charMap['\\']=89; charMap['\n']=90; charMap['\'']=91; charMap['\"']=92; 
  }
  return charMap;
}

//----------------------------------------------------------------------------//
//! Converts a character according to ALPHABET
//! @param  c  character to be mapped
//----------------------------------------------------------------------------//
static char charConvert(const char c,
    const unsigned int alph)
{

  assert((alph == 0 || alph == 1 || alph == 2) && "alph must be in {0,1,2}");
  switch (alph) {
    case 0: if (isalpha(c)) {return tolower(c);} break;
    case 1: if (isalpha(c)) {return c;} break;
    case 2: if (c != EOF) {return c;} break;
  }
  return strdup("")[0];
}

//----------------------------------------------------------------------------//
//! Count how many valid characters are contained in a file
//! @param  fileName  pointer for filename to be read
//----------------------------------------------------------------------------//
static size_t getFileLen(const char *fileName,
    const unsigned int alph)
{

  checkFileName(fileName);
  FILE *fCount = fopen(fileName, "r");
  unsigned int count = 0;
  char c, tmp;

  while (c != EOF) {
    c = fgetc(fCount);
    tmp = charConvert(c, alph);
    if (tmp != strdup("")[0]) count++;
  }

  fclose(fCount);
  return count;
}

//----------------------------------------------------------------------------//
//! Write ints and chars to arrays used in the calculation
//! @param  fileName  pointer for filename to be read
//! @param  fileChars  write valid characters here
//! @param  fileUints  write valid uints here
//----------------------------------------------------------------------------//
static void readFile(const char *fileName,
    const unsigned int alph,
    char *fileChars,
    unsigned int *fileUints)
{

  checkFileName(fileName);
  FILE *fCount = fopen(fileName, "r");
  unsigned int count = 0;
  char c, tmp;

  std::unordered_map<char, unsigned int> charMap = charMapInit(alph);
  while (c != EOF) {
    c = fgetc(fCount);
    tmp = charConvert(c, alph);
    if (tmp != strdup("")[0]) {
      fileUints[count] = charMap[tmp];
      fileChars[count] = tmp;
      count++;
    }
  }
  fclose(fCount);
}

//----------------------------------------------------------------------------//
//! Replaces "\n" with "\\" so that output files are neat
//! @param  string  string to be converted
//! @param  strLen  length of string 
//----------------------------------------------------------------------------//
static char *remSlash(const char *string,
    const size_t strLen)
{

  char *out = (char *)malloc((strLen+1)*sizeof(char));
  for(unsigned int i = 0; i < strLen; i++) {
    if (string[i] == '\n') {memcpy(&out[i],strdup("\\"),sizeof(char));}
    else {out[i] = string[i];}
  }
  out[strLen] = '\0';

  return out;
}

//----------------------------------------------------------------------------//
//! Initializing function: make io struct to simplify output
//! @param  outPre  prefix for output file
//! @param  fileName  file to be randomly generated
//! @param  inpDir  directory for input file
//! @param  outDir  directory for output file
//! @param  alph  which alphabet to use, determines twSize
//----------------------------------------------------------------------------//
struct ioPar ioInit(const char *outPre,
    const char *fileName,
    const char *inpDir,
    const char *outDir,
    const unsigned int alph)
{

  // format input file name and initialize output
  char inpName[128];
  unsigned int twSize;
  switch(alph) {
    case 0: twSize = 26; break;
    case 1: twSize = 52; break;
    case 2: twSize = 93; break;
  }
  if (!strcmp(inpDir,"")) {strcpy(inpName, fileName);}
  else {strcpy(inpName, inpDir); strcat(inpName, fileName);}
  struct ioPar var = {fileName, twSize, getFileLen(inpName, alph)};

  // setup output file
  char outName[64];
  strcpy(outName, outDir); strcat(outName, outPre); strcat(outName, var.fName);
  var.fOut = fopen(outName, "w");

  // copy data from input to array members
  var.fUints = (unsigned int *)malloc(var.numChars*sizeof(unsigned int));
  var.fChars = (char *)malloc((var.numChars+1)*sizeof(char));
  readFile(inpName, alph, var.fChars, var.fUints);
  return var;
}

//----------------------------------------------------------------------------//
//! Print header to output file
//! @param  io  struct with io data
//----------------------------------------------------------------------------//
void ioHeader(struct ioPar &io,
    const bool gpu)
{
  time_t currentTime = time(NULL);
  fprintf(io.fOut, "%s", ctime(&currentTime));
  fprintf(io.fOut, "%s has %u valid characters\n", io.fName,
      (unsigned int)io.numChars);
  fprintf(io.fOut, "Using a %2u character alphabet\n", io.twSize);
  if (gpu) {fprintf(io.fOut, "Translating %s with:\n", io.fName);}
  else {fprintf(io.fOut, "Translating %s with CPU\n", io.fName);}
}

//----------------------------------------------------------------------------//
//! Print device info to output file
//! @param  io  struct with io data
//! @param  devId  device number
//! @param  devName  name of device
//----------------------------------------------------------------------------//
void ioDevInfo(struct ioPar &io,
    const unsigned int devId,
    const char *devName)
{
  fprintf(io.fOut, "%u \"%s\"\n", devId, devName);
}

//----------------------------------------------------------------------------//
//! Print titles of output columns
//! @param  io  struct with io data
//! @param  gpu  if false, the cpu was used in matching the target
//----------------------------------------------------------------------------//
void ioColTitles(struct ioPar &io,
    const bool gpu)
{
  if (gpu) {fprintf(io.fOut, "\ngpuID     TargetString     NumsGenerated\n");}
  else {fprintf(io.fOut, "\ncpuOnly   TargetString     NumsGenerated\n");}
}

//----------------------------------------------------------------------------//
//! If a target was matched, write info about matching it to output file
//! @param  io  struct with io data
//! @param  targStr  target that was matched
//! @param  numsSoFar  how many numbers were generated in matching target
//! @param  devId  device number
//! @param  gpu  if false, the cpu was used in matching the target
//----------------------------------------------------------------------------//
void ioWord(struct ioPar &io,
    const char *targStr,
    const unsigned int targLen,
    const unsigned int numsSoFar,
    const unsigned int devId,
    const bool gpu)
{
  char *outString = remSlash(targStr, targLen);
  if (gpu) {
    fprintf(io.fOut, "%u             %8s      %12u\n",
        devId, outString, numsSoFar);
  }
  else {
    fprintf(io.fOut, "%s             %8s      %12u\n",
        "c", outString, numsSoFar);
  }
  free(outString);
}

//----------------------------------------------------------------------------//
//! Print footer to output file
//! @param  io  struct with io data
//! @param  elapsed  time elapsed over iterations
//----------------------------------------------------------------------------//
void ioFooter(struct ioPar &io,
    const double elapsed)
{
  time_t currentTime = time(NULL);
  fprintf(io.fOut, "\nGenerating %s took %.3f seconds \n",
      io.fName, elapsed);
  currentTime = time(NULL);
  fprintf(io.fOut, "%s", ctime(&currentTime));
}

//----------------------------------------------------------------------------//
//! Free mem of io members and close its file member
//! @param  io  struct with io data
//----------------------------------------------------------------------------//
void ioFree(struct ioPar &io)
{
  free(io.fChars);
  free(io.fUints);
  fclose(io.fOut);
}
