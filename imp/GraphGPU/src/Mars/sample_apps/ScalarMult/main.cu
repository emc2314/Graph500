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
 * ScalarMult application
 ******************************************************************/

#include "MarsInc.h"
#include "global.h"

#define DST_SIZE 20

#define __OUTPUT__

FILE *smult_out;

void printFun(void* key, void* val, int keySize, int valSize)
{
	int* k = (int*)key;
	float* v = (float*)val;
#ifdef __OUTPUT__	  
	printf("%d\t%f\n", *k, *v);
#endif
	fprintf(smult_out, "%d\t%E\n", *k, *v);
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

  // key
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
  // val
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
	spec->workflow = MAP_GROUP;
	//1 for outputing result to Host
	//by default, Mars keeps results in device memory
	spec->outputToHost = 1;

	//----------------------------------------------
	//preprocess
	//----------------------------------------------
	int key;
	float val;

	smult_out = fopen(argv[2], "w");

	// // scalar
	// FILE *sf = fopen(argv[3], "r");
	// fseek(sf, 0, SEEK_END);
	// int fileSize_s = ftell(sf) + 1;
	// rewind(sf);
	// char* h_filebuf_s = (char*)malloc(fileSize_s); 
	// fread(h_filebuf_s, fileSize_s, 1, sf);
	// fclose(sf);
	// mysplit(strtok(h_filebuf_s, "\n"), '\t', &key, &val);	
	// float scalar = val;
	// float mixing_c = atof(argv[4]);
	// float s = (1.0 - mixing_c) / scalar;
	float s = atof(argv[3]);

#ifdef _DEBUG
	printf("s = %f\n", s);
#endif
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
	if( 0 <= key )
	  AddMapInputRecord(spec, &key, &val, sizeof(int), sizeof(float));
	while( p_q != NULL) {
	  p_q = strtok(NULL, "\n");
	  if( p_q != NULL) {
	    mysplit(p_q, '\t', &key, &val);
	    if( 0 <= key )
	      AddMapInputRecord(spec, &key, &val, sizeof(int), sizeof(float));
	  }
	}
	
	endTimer("Preprocess", &preprocessTimer);

	//----------------------------------------------
	//start mapreduce
	//----------------------------------------------
	MapReduce(spec, s);

	startTimer(&postprocessTimer);
	
	//----------------------------------------------
	//further processing
	//----------------------------------------------
	Output(spec, spec->outputRecordCount);


	//----------------------------------------------
	//finish
	//----------------------------------------------
        fclose(smult_out);
	FinishMapReduce(spec);

	free(h_filebuf_q);

	endTimer("Postprocess", &postprocessTimer);
	endTimer("Total", &totalTimer);

	return 0;
}
