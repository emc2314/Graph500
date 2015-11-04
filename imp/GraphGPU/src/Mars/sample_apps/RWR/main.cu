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
 * Random Walk with Restart application
 ******************************************************************/

#include "MarsInc.h"
#include "global.h"

#define DST_SIZE 20

//#define __OUTPUT__

FILE *rwr_out;

void printFun(void* key, void* val, int keySize, int valSize)
{
	int* k = (int*)key;
	float v = ((RWR_VAL_T*)val)->dst;
	bool is_v = ((RWR_VAL_T*)val)->is_v;
	if(is_v == true) {
#ifdef __OUTPUT__
	  printf("%d\tv%E\n", *k, v);
#endif
	  fprintf(rwr_out, "%d\tv%E\n", *k, v);
	}
	else {
#ifdef __OUTPUT__	  
	  printf("%d\t%E\n", *k, v);
#endif
	  fprintf(rwr_out, "%d\t%E\n", *k, v);
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

int mysplit(char *p, char split, RWR_KEY_T *key, RWR_VAL_T *val) {
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
	spec->numMapReduce = 2;

	//1 for outputing result to Host
	//by default, Mars keeps results in device memory
	spec->outputToHost = 1;

	if(argc < 7) {
	  printf("Usage: PageRank in_M in_v pr_out num_node\n");
	  exit(1);
	}

	//----------------------------------------------
	//preprocess
	//----------------------------------------------

	rwr_out = fopen(argv[3], "w");	
	int num_node = atoi(argv[4]);
	float mixing_c = atof(argv[5]);
	int niteration = atoi(argv[6]);

	spec->argi = num_node;
	spec->argf = mixing_c;

	// for M
	FILE* fp = fopen(argv[1], "r");
	RWR_KEY_T key;
	RWR_VAL_T val;

	fseek(fp, 0, SEEK_END);
        int fileSize = ftell(fp) + 1;
        rewind(fp);
        char* h_filebuf = (char*)malloc(fileSize);
        fread(h_filebuf, fileSize, 1, fp);
        fclose(fp);

	char *p = h_filebuf;
	p = strtok(p, "\n");
	if( mysplit(p, '\t', &key, &val) != 0 ) 
	  AddMapInputRecord(spec, &key, &val, sizeof(RWR_KEY_T), sizeof(RWR_VAL_T));
	while( p != NULL) {
	  p = strtok(NULL, "\n");
	  if( p != NULL) {
	    if( mysplit(p, '\t', &key, &val) != 0)
	      AddMapInputRecord(spec, &key, &val, sizeof(RWR_KEY_T), sizeof(RWR_VAL_T));
	  }
	}

	// for v
	char* rwr_in = argv[2];
	FILE *rwr_init_v;
	if(niteration == 1) {
	  puts("generating rwr_init_vector...");
	  FILE *rwr_init_v = fopen(rwr_in, "w");
	  float initial_rank = 1.0 / (float)num_node;
	  for(int i = 0; i < num_node; i++) {
	    fprintf(rwr_init_v, "%d\tv%E\n", i, initial_rank);
	  }
	  fclose(rwr_init_v);
	}

	rwr_init_v = fopen(rwr_in, "r");
	fseek(rwr_init_v, 0, SEEK_END);
	int fileSize_v = ftell(rwr_init_v) + 1;
	rewind(rwr_init_v);
	char* h_filebuf_v = (char*)malloc(fileSize_v); 
	fread(h_filebuf_v, fileSize_v, 1, rwr_init_v);
	fclose(rwr_init_v);

	char *p_v = h_filebuf_v;	
	p_v = strtok(p_v, "\n");	
	mysplit(p_v, '\t', &key, &val);
	if( 0 <= key.src )
	  AddMapInputRecord(spec, &key, &val, sizeof(RWR_KEY_T), sizeof(RWR_VAL_T));
	while( p_v != NULL) {
	  p_v = strtok(NULL, "\n");
	  if( p_v != NULL) {
	    mysplit(p_v, '\t', &key, &val);
	    if( 0 <= key.src )
	      AddMapInputRecord(spec, &key, &val, sizeof(RWR_KEY_T), sizeof(RWR_VAL_T));
	  }
	}
	
	endTimer("Preprocess", &preprocessTimer);

	//----------------------------------------------
	//start mapreduce
	//----------------------------------------------
	MapReduce(spec);

	startTimer(&postprocessTimer);
	
	//----------------------------------------------
	//further processing
	//----------------------------------------------
	Output(spec, spec->outputRecordCount);


	//----------------------------------------------
	//finish
	//----------------------------------------------
	fclose(rwr_out);
	FinishMapReduce(spec);

	free(h_filebuf);
	free(h_filebuf_v);

	endTimer("Postprocess", &postprocessTimer);
	endTimer("Total", &totalTimer);

	return 0;
}
