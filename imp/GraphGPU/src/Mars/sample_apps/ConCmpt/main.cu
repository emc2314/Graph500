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
 * Connected Component application
 ******************************************************************/

#include "MarsInc.h"
#include "global.h"

#define DST_SIZE 20

//#define __OUTPUT__
//#define _DEBUG_MAIN

FILE *cc_out;

void printFun(void* key, void* val, int keySize, int valSize)
{
	int* k = (int*)key;
	int v = ((CC_VAL_T*)val)->dst;
	bool is_v = ((CC_VAL_T*)val)->is_v;

	if(*k != -1 && v != -1) {
	  if(is_v == true) {
#ifdef __OUTPUT__
	    printf("%d\tv%d\n", *k, v);
#endif
	    fprintf(cc_out, "%d\tv%d\n", *k, v);
	  }
	  else {
#ifdef __OUTPUT__	  
	    printf("%d\t%d\n", *k, v);
#endif
	    fprintf(cc_out, "%d\t%d\n", *k, v);
	  }
	}
}


// void printFun(void* key, void* val, int keySize, int valSize)
// {
//   int* k = (int*)key;
//   int* v = (int*)val;
//   if(*k != -1 && *v != -1) {
// #ifdef __OUTPUT__
//     printf("%d\t%d\n", *k, *v);
// #endif
//     fprintf(cc_out, "%d\t%d\n", *k, *v);
//   }
// }

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

int mysplit(char *p, char split, CC_KEY_T *key, CC_VAL_T *val) {
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

void gen_one_file(int number_nodes, int start_pos, int len, FILE* curbm)
{
  int j = 0;
  int count = 0;
  fprintf(curbm, "# component vector file - mars\n");
  fprintf(curbm, "# number of nodes in graph = %d, start_pos = %d\n",
	  number_nodes, start_pos);
  printf("creating bitmask generation cmd for node %d ~ %d\n",
	 start_pos, start_pos+len);
  
  for(int i = 0; i < number_nodes; i++) {
    int cur_nodeid = start_pos + i;
    fprintf(curbm, "%d\tv%d\n", cur_nodeid, cur_nodeid);
    if(++j > len/10) {
      printf(".");
      j = 0;
    }
    if(++count >= len)
      break;
  }
  printf("\n");
}

void gen_component_vector_file(int number_nodes, FILE* curbm)
{
  //int start_pos = 0;
  int max_filesize = 10000000;

  for(int i = 0; i < number_nodes; i += max_filesize) {
    int len = max_filesize;
    if(len > number_nodes - i)
      len = number_nodes - i;
    gen_one_file(number_nodes, i, len, curbm);
  }
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

  //----------------------------------------------
  //preprocess
  //----------------------------------------------
  if(argc < 5) {
    printf("Usage: XXX\n");
    exit(1);
  }

  char *cc_in = argv[2];
  char *cc_out_name = argv[3];
  cc_out = fopen(cc_out_name, "w");	
  //char *convergence_check = argv[4];
  int num_node = atoi(argv[4]);
  int niteration = atoi(argv[5]);

  CC_KEY_T key;
  CC_VAL_T val;

  // for M
  FILE* fp = fopen(argv[1], "r");
  fseek(fp, 0, SEEK_END);
  int fileSize = ftell(fp) + 1;
  rewind(fp);
  char* h_filebuf = (char*)malloc(fileSize);
  fread(h_filebuf, fileSize, 1, fp);
  fclose(fp);

  char *p = h_filebuf;
  p = strtok(p, "\n");
  if( mysplit(p, '\t', &key, &val) != 0 ) {
    val.is_v = false;
    AddMapInputRecord(spec, &key, &val, sizeof(CC_KEY_T), sizeof(CC_VAL_T));    
  }
  while( p != NULL) {
    p = strtok(NULL, "\n");
    if( p != NULL) {
      if( mysplit(p, '\t', &key, &val) != 0) {
	val.is_v = false;
	AddMapInputRecord(spec, &key, &val, sizeof(CC_KEY_T), sizeof(CC_VAL_T));
      }
    }
  }

  // for v
  FILE *curbm;
  if(niteration == 1) {
    puts("generating cc_init_vector...");
    curbm = fopen(cc_in, "w");
    gen_component_vector_file(num_node, curbm);
    fclose(curbm);
  }
  
  curbm = fopen(cc_in, "r");
  fseek(curbm, 0, SEEK_END);
  int fileSize_v = ftell(curbm) + 1;
  rewind(curbm);
  char* h_filebuf_v = (char*)malloc(fileSize_v); 
  fread(h_filebuf_v, fileSize_v, 1, curbm);
  fclose(curbm);

  char *p_v = h_filebuf_v;	
  p_v = strtok(p_v, "\n");
  if(mysplit(p_v, '\t', &key, &val) != 0) {
    if( 0 <= key.src ) {
      val.is_v = true;
      AddMapInputRecord(spec, &key, &val, sizeof(CC_KEY_T), sizeof(CC_VAL_T));
    }
  }
  while( p_v != NULL) {
    p_v = strtok(NULL, "\n");
    if( p_v != NULL) {
      if(mysplit(p_v, '\t', &key, &val) != 0) {
	if( 0 <= key.src ) {
	  val.is_v = true;
	  AddMapInputRecord(spec, &key, &val, sizeof(CC_KEY_T), sizeof(CC_VAL_T));
	}
      }
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
  fclose(cc_out);

  //----------------------------------------------
  //finish
  //----------------------------------------------
  FinishMapReduce(spec);
  
  free(h_filebuf);
  free(h_filebuf_v);

  //CHECK CONVERGENCE
//   int num_changed = 0;
//   int num_unchanged = 0;
//   int size = 16;
//   char s[size];
//   cc_out = fopen(cc_out_name, "r");
//   if(fgets(s, size, cc_out) != NULL) {
//     mysplit(s, '\t', &num_unchanged);
//   }
//   if(fgets(s, size, cc_out) != NULL) {
//     mysplit(s, '\t', &num_changed);
//   }
//   printf("num_changed = %d, num_unchanged = %d\n", num_changed, num_unchanged);

//   if(num_changed == 0) {
//     FILE *outf = fopen(convergence_check, "w");
//     fprintf(outf, "CONVERGED");
//     fclose(outf);
//   }
  
  endTimer("Postprocess", &postprocessTimer);
  endTimer("Total", &totalTimer);
  
  return 0;
}
