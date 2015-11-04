/***********************************************************************
 	graphgpu
	Authors: Koichi Shirahata, Hitoshi Sato, Toyotaro Suzumura, and Satoshi Matsuoka

This software is licensed under Apache License, Version 2.0 (the  "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
***********************************************************************/

/******************************************************************
 * Connected Component Step 3 
 ******************************************************************/

#include "MarsInc.h"
#include "global.h"

#define DST_SIZE 20

//#define __OUTPUT__

FILE *cc3_out;

void printFun(void* key, void* val, int keySize, int valSize)
{
	int* k = (int*)key;
	int* v = (int*)val;
	if(*k != -1 && *v != -1) {
#ifdef __OUTPUT__
	  printf("%d\t%d\n", *k, *v);
#endif
	  fprintf(cc3_out, "%d\t%d\n", *k, *v);
	}
}

void Output(Spec_t* spec, int num)
{
	if (spec->outputToHost != 1)
	{
		printf("Error: please set outputToHost to 1 first!\n");
		return;
	}
	if (num > spec->outputRecordCount) num = spec->outputRecordCount;

	PrintOutputRecords(spec, num, printFun);	
}

int mysplit(char *p, char split, CC3_KEY_T *key, CC3_VAL_T *val) {
  int i = 0;
  char str[DST_SIZE];

  key->src = -1;
  val->dst = -1;
  
  if(*p == '#')
    return 0;

  // key->src
  for(i = 0; (unsigned char)(*p - '0') <= 9 || *p == '-'; i++, p++) {
    str[i] = *p;
  }
  str[i] = '\0';
  key->src = atoi(str);

  if( i <= 0) {
    puts("Error in format of your edge file");
    exit(-1);
  }

  // tab
  while(*p == split)
    p++;
  // val->is_v
  if(*p == 'v') {
    val->is_v = true;
    p++;
  } else {
    val->is_v = false;
  }
  // val->dst
  for(i = 0; (unsigned char)(*p - '0') <= 9 || *p == '.' || *p == 'E' || *p == 'e' || *p == '+' || *p == '-'; i++, p++) {
    str[i] = *p;
  }
  str[i] = '\0';
  val->dst = atoi(str);

  return 1;
}

int mysplit(char *p, char split, int *val) {
  int i = 0;
  char str[DST_SIZE];

  *val = -1;
  
  if(*p == '#')
    return 0;

  // key->src
  for(i = 0; (unsigned char)(*p - '0') <= 9 || *p == '-'; i++, p++)
    ;
  if( i <= 0) {
    puts("Error in format of your edge file");
    exit(-1);
  }

  // tab
  while(*p == split)
    p++;
  // val
  for(i = 0; (unsigned char)(*p - '0') <= 9; i++, p++) {
    str[i] = *p;
  }
  str[i] = '\0';
  *val = atoi(str);

  return 1;
}

int main( int argc, char** argv) 
{
  TimeVal_t totalTimer;
  TimeVal_t preprocessTimer;
  TimeVal_t postprocessTimer;

  startTimer(&totalTimer);
  startTimer(&preprocessTimer);

  Spec_t *spec = GetDefaultSpec();
  //MAP_ONLY, MAP_GROUP, or MAP_REDUCE
  spec->workflow = MAP_REDUCE;
  //1 for outputing result to Host
  //by default, Mars keeps results in device memory
  spec->outputToHost = 1;

  //----------------------------------------------
  //preprocess
  //----------------------------------------------
  if(argc < 3){
    printf("Usage: XXX\n");
  }
  cc3_out = fopen(argv[2], "w");

  // for curbm
  FILE* fp = fopen(argv[1], "r");
  CC3_KEY_T key;
  CC3_VAL_T val;

  fseek(fp, 0, SEEK_END);
  int fileSize = ftell(fp) + 1;
  rewind(fp);
  char* h_filebuf = (char*)malloc(fileSize);
  fread(h_filebuf, fileSize, 1, fp);
  fclose(fp);

  char *p = h_filebuf;
  p = strtok(p, "\n");
  if( mysplit(p, '\t', &key, &val) != 0 ) {
    AddMapInputRecord(spec, &key, &val, sizeof(CC3_KEY_T), sizeof(CC3_VAL_T));
  }
  while( p != NULL) {
    p = strtok(NULL, "\n");
    if( p != NULL) {
      if( mysplit(p, '\t', &key, &val) != 0) {
	AddMapInputRecord(spec, &key, &val, sizeof(CC3_KEY_T), sizeof(CC3_VAL_T));
      }
    }
  }
  endTimer("Preprocess", &preprocessTimer);

  //----------------------------------------------
  //start mapreduce
  //----------------------------------------------
  MapReduce(spec);

  //----------------------------------------------
  //further processing
  //----------------------------------------------
  startTimer(&postprocessTimer);
  Output(spec, spec->outputRecordCount);
  
  //----------------------------------------------
  //finish
  //----------------------------------------------
  fclose(cc3_out);
  FinishMapReduce(spec);

  free(h_filebuf);

  int num_changed = 0;
  int num_unchanged = 0;
  char s[16];
  cc3_out = fopen(argv[2], "r");
  if(fgets(s, 16, cc3_out) != NULL) {
    mysplit(s, '\t', &num_unchanged);
  }
  if(fgets(s, 16, cc3_out) != NULL) {
    mysplit(s, '\t', &num_changed);
  }
  printf("num_changed = %d, num_unchanged = %d\n", num_changed, num_unchanged);

  if(num_changed == 0) {
    FILE *outf = fopen(argv[3], "w");
    fprintf(outf, "CONVERGED");
    fclose(outf);
  }

  endTimer("Postprocess", &postprocessTimer);
  endTimer("Total", &totalTimer);

  return 0;
}
