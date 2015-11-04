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
 * PageRank application
 ******************************************************************/

#include "MarsInc.h"
#include "global.h"

#define DST_SIZE 20

// #define __OUTPUT__

FILE *pr_out;
//FILE *pr2_result;
int converged_reducer = 0;

void printFun(void* key, void* val, int keySize, int valSize)
{
	int* k = (int*)key;
	float v = ((PR_VAL_T*)val)->dst;
	bool is_v = ((PR_VAL_T*)val)->is_v;
	if(*k >= 0) {
	  if(is_v == true) {
#ifdef __OUTPUT__
	    printf("%d\tv%E\n", *k, v);
#endif
	    fprintf(pr_out, "%d\tv%E\n", *k, v);
	  }
	  else {
#ifdef __OUTPUT__	  
	    printf("%d\t%E\n", *k, v);
#endif
	    fprintf(pr_out, "%d\t%E\n", *k, v);
	    //fprintf(stdout, "%d\t%E\n", *k, v);
	  }
 	}
 	else if(*k == -1 && v == 0.0f) {
 	  converged_reducer++;
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

int mysplit(char *p, char split, PR_KEY_T *key, PR_VAL_T *val) {
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

  TimeVal_t addMapTimer;

  if(argc < 5) {
    printf("Usage: PageRank in_M in_v pr_out num_nodes\n");
    exit(1);
  }

  startTimer(&totalTimer);
  startTimer(&preprocessTimer);

  Spec_t *spec = GetDefaultSpec();
	//MAP_ONLY, MAP_GROUP, or MAP_REDUCE
	spec->workflow = MAP_REDUCE;
	//1 for outputing result to Host
	//by default, Mars keeps results in device memory
	spec->outputToHost = 1;
	spec->numMapReduce = 2;

	// MPI_Init(&argc, &argv);

 	// MPI_Comm_size(MPI_COMM_WORLD, &spec->numProcs);
 	// MPI_Comm_rank(MPI_COMM_WORLD, &spec->rank);

	// printf("%d, %d\n", spec->rank, spec->numProcs);



	
	//----------------------------------------------
	//preprocess
	//----------------------------------------------


	int niter = atoi(argv[5]);
	pr_out = fopen(argv[3], "w");	
	int num_nodes = atoi(argv[4]);
	float mixing_c = 0.85;

	
	spec->argi = num_nodes;
	spec->argf = mixing_c;


	char* h_filebuf;
	char* h_filebuf_v;
	
	// if(spec->rank == 0) {


	// for M
	FILE* fp = fopen(argv[1], "r");
	PR_KEY_T key;
	PR_VAL_T val;

	fseek(fp, 0, SEEK_END);
        int fileSize = ftell(fp) + 1;
        rewind(fp);
        h_filebuf = (char*)malloc(fileSize);
        fread(h_filebuf, fileSize, 1, fp);
        fclose(fp);

	char *p = h_filebuf;

	startTimer(&addMapTimer);

	p = strtok(p, "\n");
	if( mysplit(p, '\t', &key, &val) != 0 ) 
	  AddMapInputRecord(spec, &key, &val, sizeof(PR_KEY_T), sizeof(PR_VAL_T));
	while( p != NULL) {
	  p = strtok(NULL, "\n");
	  if( p != NULL) {
	    if( mysplit(p, '\t', &key, &val) != 0)
	      AddMapInputRecord(spec, &key, &val, sizeof(PR_KEY_T), sizeof(PR_VAL_T));
	  }
	}


	// for v
	FILE *pr_in;
	//	if(strcmp(argv[2], "pagerank_init_vector") == 0) {
	if(niter == 1) {
	  puts("generating pagerank_init_vector...");
	  pr_in = fopen(argv[2], "w");
	  float initial_rank = 1.0 / (float)num_nodes;
	  for(int i = 0; i < num_nodes; i++) {
	    fprintf(pr_in, "%d\tv%E\n", i, initial_rank);
	  }

	  fclose(pr_in);
	}

	pr_in = fopen(argv[2], "r");
	fseek(pr_in, 0, SEEK_END);
	int fileSize_v = ftell(pr_in) + 1;
	rewind(pr_in);
	h_filebuf_v = (char*)malloc(fileSize_v); 
	fread(h_filebuf_v, fileSize_v, 1, pr_in);
	fclose(pr_in);

	char *p_v = h_filebuf_v;	

	p_v = strtok(p_v, "\n");	
	mysplit(p_v, '\t', &key, &val);
	if( 0 <= key.src )
	  AddMapInputRecord(spec, &key, &val, sizeof(PR_KEY_T), sizeof(PR_VAL_T));
	while( p_v != NULL) {
	  p_v = strtok(NULL, "\n");
	  if( p_v != NULL) {
	    mysplit(p_v, '\t', &key, &val);
	    if( 0 <= key.src )
	      AddMapInputRecord(spec, &key, &val, sizeof(PR_KEY_T), sizeof(PR_VAL_T));
	  }
	}
	
	endTimer("Preprocess", &preprocessTimer);
	endTimer("AddMap", &addMapTimer);

	// }


	
	//----------------------------------------------
	//start mapreduce
	//----------------------------------------------
	MapReduce(spec);

	
	//----------------------------------------------
	//further processing
	//----------------------------------------------
	// if(spec->rank == 0) {

	startTimer(&postprocessTimer);
	
	Output(spec, spec->outputRecordCount);
	printf("converged_reducer : %d / %d\n", converged_reducer, num_nodes);
	//printf("%d\n", spec->outputRecordCount);

	FILE *outf = fopen("converged_reducer", "w");
	if(converged_reducer >= num_nodes)
	  fprintf(outf, "CONVERGED");
	fclose(outf);

	//----------------------------------------------
	//finish
	//----------------------------------------------
        fclose(pr_out);
	FinishMapReduce(spec);

	
	free(h_filebuf);
	free(h_filebuf_v);

	endTimer("Postprocess", &postprocessTimer);
	endTimer("Total", &totalTimer);


	// }

	// MPI_Finalize();

	
	return 0;
}
