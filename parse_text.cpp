// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>

// includes, project
#include "config.h"
#include "parse_text.h"

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
static std::unordered_map<char, unsigned int> charMap
#if ALPHABET == 0
  {
    {'a', 0},{'b', 1},{'c', 2},{'d', 3},{'e', 4},{'f', 5},{'g', 6},{'h', 7},
    {'i', 8},{'j', 9},{'k',10},{'l',11},{'m',12},{'n',13},{'o',14},{'p',15},
    {'q',16},{'r',17},{'s',18},{'t',19},{'u',20},{'v',21},{'w',22},{'x',23},
    {'y',24},{'z',25} 
  };

#elif ALPHABET == 1
  {
    {'a', 0},{'b', 1},{'c', 2},{'d', 3},{'e', 4},{'f', 5},{'g', 6},{'h', 7},
    {'i', 8},{'j', 9},{'k',10},{'l',11},{'m',12},{'n',13},{'o',14},{'p',15},
    {'q',16},{'r',17},{'s',18},{'t',19},{'u',20},{'v',21},{'w',22},{'x',23},
    {'y',24},{'z',25},
    {'A',26},{'B',27},{'C',28},{'D',29},{'E',30},{'F',31},{'G',32},{'H',33},
    {'I',34},{'J',35},{'K',36},{'L',37},{'M',38},{'N',39},{'O',40},{'P',41},
    {'Q',42},{'R',43},{'S',44},{'T',45},{'U',46},{'V',47},{'W',48},{'X',49},
    {'Y',50},{'Z',51}
  };

#elif ALPHABET == 2
  {
    {'a', 0},{'b', 1},{'c', 2},{'d', 3},{'e', 4},{'f', 5},{'g', 6},{'h', 7},
    {'i', 8},{'j', 9},{'k',10},{'l',11},{'m',12},{'n',13},{'o',14},{'p',15},
    {'q',16},{'r',17},{'s',18},{'t',19},{'u',20},{'v',21},{'w',22},{'x',23},
    {'y',24},{'z',25},
    {'A',26},{'B',27},{'C',28},{'D',29},{'E',30},{'F',31},{'G',32},{'H',33},
    {'I',34},{'J',35},{'K',36},{'L',37},{'M',38},{'N',39},{'O',40},{'P',41},
    {'Q',42},{'R',43},{'S',44},{'T',45},{'U',46},{'V',47},{'W',48},{'X',49},
    {'Y',50},{'Z',51},
    {'0',52},{'1',53},{'2',54},{'3',55},{'4',56},{'5',57},{'6',58},{'7',59},
    {'8',60},{'9',61},
    {' ',62},{',',63},{';',64},{':',65},{'.',66},{'!',67},{'?',68},{'#',69},
    {'$',70},{'%',71},{'^',72},{'&',73},{'_',74},{'+',75},{'-',76},{'*',77},
    {'/',78},{'=',79},{'|',80},{'(',81},{')',82},{'[',83},{']',84},{'{',85},
    {'}',86},{'<',87},{'>',88},
    {'\\',89},{'\n',90},{'\'',91},{'\"',92},{'\@',94}
  };

#endif

//----------------------------------------------------------------------------//
//! Converts a character according to ALPHABET
//! @param  c  character to be mapped
//----------------------------------------------------------------------------//
static char charConvert(const char c)
{

#if ALPHABET == 0
    if (isalpha(c)) {return tolower(c);}
#elif ALPHABET == 1
    if (isalpha(c)) {return c;}
#elif ALPHABET == 2
    if (c != EOF) {return c;}
#endif
    else {return strdup("")[0];}

}

//----------------------------------------------------------------------------//
//! Count how many valid characters are contained in a file
//! @param  fileName  pointer for filename to be read
//----------------------------------------------------------------------------//
size_t getFileLen(const char *fileName)
{

  checkFileName(fileName);
  FILE *fCount = fopen(fileName, "r");
  unsigned int count = 0;
  char c, tmp;

  while (c != EOF) {
    c = fgetc(fCount);
    tmp = charConvert(c);
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
void readFile(const char *fileName,
    char *fileChars,
    unsigned int *fileUints)
{

  checkFileName(fileName);
  FILE *fCount = fopen(fileName, "r");
  unsigned int count = 0;
  char c, tmp;

  while (c != EOF) {
    c = fgetc(fCount);
    tmp = charConvert(c);
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
//! @param  stringLen  length of string 
//----------------------------------------------------------------------------//
char *removeSlash(const char *string,
    const size_t stringLen)
{

  char *out = (char *)malloc((stringLen+1)*sizeof(char));
  for(unsigned int i = 0; i < stringLen; i++) {
    if (string[i] == '\n') {memcpy(&out[i],strdup("\\"),sizeof(char));}
    else {out[i] = string[i];}
  }
  out[stringLen] = '\0';

  return out;

}
