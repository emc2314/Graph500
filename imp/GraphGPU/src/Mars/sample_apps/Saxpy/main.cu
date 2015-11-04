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
 * Saxpy application
 ******************************************************************/

#include "MarsInc.h"
#include "global.h"

#define DST_SIZE 20

//#define __OUTPUT__

FILE *saxpy_out;

void printFun(void* key, void* val, int keySize, int valSize)
{
	int* k = (int*)key;
	// float v = ((SAXPY_VAL_T*)val)->dst;
	// bool is_v = ((SAXPY_VAL_T*)val)->is_v;
	float v = *(float*)val;
	// bool is_v = ((SAXPY_VAL_T*)val)->is_v;
// 	if(is_v == true) {
// #ifdef __OUTPUT__
// 	  printf("%d\tv%f\n", *k, v);
// #endif
// 	  if(*k != -1)
// 	    fprintf(saxpy_out, "%d\tv%E\n", *k, v);
// 	}
// 	else {
// #ifdef __OUTPUT__	  
// 	  printf("%d\t%f\n", *k, v);
// #endif
// 	  if(*k != -1)
// 	    fprintf(saxpy_out, "%d\t%E\n", *k, v);
// 	}

#ifdef __OUTPUT__
	  printf("%d\tv%f\n", *k, v);
#endif
	  if(*k != -1)
	    fprintf(saxpy_out, "%d\tv%E\n", *k, v);

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

int mysplit(char *p, char split, SAXPY_KEY_T *key, SAXPY_VAL_T *val, bool is_y) {
  int i = 0;
  char str[DST_SIZE];

  key->src = -1;
  val->dst = -1;
  
  if(*p == '#')
    return 0;

  // // key->is_y
  // key->is_y = is_y;


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
    // val->is_y = true;
    p++;
  } else {
    // val->is_y = false;
  }

  val->is_y = is_y;

  // val->dst
  for(i = 0; (unsigned char)(*p - '0') <= 9 || *p == '.' || *p == 'E' || *p == 'e' || *p == '+' || *p == '-'; i++, p++) {
    str[i] = *p;
  }
  str[i] = '\0';
  val->dst = atof(str);

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
	SAXPY_KEY_T key;
	SAXPY_VAL_T val;

	// input alpha
	// spec->farg = atof(argv[3]);
	// saxpy_out = fopen(argv[4], "w");
	saxpy_out = fopen(argv[3], "w");
	spec->farg = atof(argv[4]);
	// float a = atof(argv[4]);
	// bool is_vector_output = atoi(argv[5]);
#if 0
	spec->barg = atoi(argv[5]);
#endif
	// for input vector (y)
	FILE* fv = fopen(argv[1], "r");
	fseek(fv, 0, SEEK_END);
        int fileSize = ftell(fv) + 1;
        rewind(fv);
        char* h_filebuf = (char*)malloc(fileSize);
        fread(h_filebuf, fileSize, 1, fv);
        fclose(fv);

	char *p = h_filebuf;

	p = strtok(p, "\n");
	if( mysplit(p, '\t', &key, &val, true) != 0 ) 
	  AddMapInputRecord(spec, &key, &val, sizeof(SAXPY_KEY_T), sizeof(SAXPY_VAL_T));
	while( p != NULL) {
	  p = strtok(NULL, "\n");
	  if( p != NULL) {
	    if( mysplit(p, '\t', &key, &val, true) != 0)
	      AddMapInputRecord(spec, &key, &val, sizeof(SAXPY_KEY_T), sizeof(SAXPY_VAL_T));
	  }
	}

	// for query (x)
	FILE *query;
	query = fopen(argv[2], "r");
	fseek(query, 0, SEEK_END);
	int fileSize_q = ftell(query) + 1;
	rewind(query);
	char* h_filebuf_q = (char*)malloc(fileSize_q); 
	fread(h_filebuf_q, fileSize_q, 1, query);
	fclose(query);

	char *p_q = h_filebuf_q;	
	p_q = strtok(p_q, "\n");	
	mysplit(p_q, '\t', &key, &val, false);
	if( 0 <= key.src )
	  AddMapInputRecord(spec, &key, &val, sizeof(SAXPY_KEY_T), sizeof(SAXPY_VAL_T));
	while( p_q != NULL) {
	  p_q = strtok(NULL, "\n");
	  if( p_q != NULL) {
	    mysplit(p_q, '\t', &key, &val, false);
	    if( 0 <= key.src )
	      AddMapInputRecord(spec, &key, &val, sizeof(SAXPY_KEY_T), sizeof(SAXPY_VAL_T));
	  }
	}
	
	endTimer("Preprocess", &preprocessTimer);

	//----------------------------------------------
	//start mapreduce
	//----------------------------------------------
	// MapReduce(spec, a, is_vector_output);
	MapReduce(spec);

	startTimer(&postprocessTimer);
	
	//----------------------------------------------
	//further processing
	//----------------------------------------------
	Output(spec, spec->outputRecordCount);


	//----------------------------------------------
	//finish
	//----------------------------------------------
        fclose(saxpy_out);
	FinishMapReduce(spec);

	free(h_filebuf);
	free(h_filebuf_q);

	endTimer("Postprocess", &postprocessTimer);
	endTimer("Total", &totalTimer);

	return 0;
}
