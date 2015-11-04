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
 * L1norm application
 ******************************************************************/

#include "MarsInc.h"
#include "global.h"

#define DST_SIZE 20

//#define __OUTPUT__
//#define _DEBUG_MAIN

FILE *l1norm_out;
float diff;


void printFun(void* key, void* val, int keySize, int valSize)
{
	int* k = (int*)key;
	float* v = (float*)val;
#ifdef __OUTPUT__	  
	printf("%d\t%E\n", *k, *v);
#endif
	fprintf(l1norm_out, "%d\t%E\n", *k, *v);
	diff = *v;
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

int mysplit(char *p, char split, int *key, float *val) {
  int i = 0;
  char str[DST_SIZE];

  // key->src = -1;
  // val->dst = -1;

  if(*p == '#')
    return 0;

  // key->src
  for(i = 0; (unsigned char)(*p - '0') <= 9 || *p == '-'; i++, p++) {
    str[i] = *p;
  }
  str[i] = '\0';
  *key = atoi(str);

  if( i <= 0) {
    puts("Error in format of your edge file");
    exit(-1);
  }

  // tab
  while(*p == split)
    p++;

  // v
  if(*p == 'v')
    p++;

  // val->dst
  for(i = 0; (unsigned char)(*p - '0') <= 9 || *p == '.' || *p == 'E' || *p == 'e' || *p == '+' || *p == '-'; i++, p++) {
    str[i] = *p;
  }
  str[i] = '\0';
  *val = atof(str);

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
	int key;
	float val;

	l1norm_out = fopen(argv[2], "w");
	diff = 1;
	float converge_threshold = -1;
	if(argv[3] != NULL) {
	  converge_threshold = atof(argv[3]);
	}

	// for query
	FILE *query;
	query = fopen(argv[1], "r");
	fseek(query, 0, SEEK_END);
	int fileSize_q = ftell(query) + 1;
	rewind(query);
	char* h_filebuf_q = (char*)malloc(fileSize_q); 
	fread(h_filebuf_q, fileSize_q, 1, query);
	fclose(query);

	char *p_q = h_filebuf_q;
	p_q = strtok(p_q, "\n");	
	mysplit(p_q, '\t', &key, &val);
#ifdef _DEBUG_MAIN
	    printf("input: key = %d, value = %f\n", key, val);
#endif
	if( 0 <= key )
	  AddMapInputRecord(spec, &key, &val, sizeof(L1N_KEY_T), sizeof(L1N_VAL_T));
	while( p_q != NULL) {
	  p_q = strtok(NULL, "\n");
	  if( p_q != NULL) {
	    mysplit(p_q, '\t', &key, &val);
#ifdef _DEBUG_MAIN
	    printf("input: key = %d, value = %f\n", key, val);
#endif
	    if( 0 <= key )
	      AddMapInputRecord(spec, &key, &val, sizeof(L1N_KEY_T), sizeof(L1N_VAL_T));
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

	if(converge_threshold != -1) {
	  FILE *outf = fopen("check_convergence", "w");
	  printf("diff = %f, converge_threshold = %f\n", diff, converge_threshold);
	  if(diff < converge_threshold)
	    fprintf(outf, "CONVERGED");
	  fclose(outf);
	}

	//----------------------------------------------
	//finish
	//----------------------------------------------
        fclose(l1norm_out);
	FinishMapReduce(spec);
	free(h_filebuf_q);

	endTimer("Postprocess", &postprocessTimer);
	endTimer("Total", &totalTimer);
	return 0;
}
