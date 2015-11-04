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

#ifndef __REDUCE_CU__
#define __REDUCE_CU__

#include "MarsInc.h"
#include "global.h"

//#define __NO_REDUCE__
// #define _DEBUG_REDUCE

__device__ void REDUCE_COUNT_FUNC//(void* key, void* vals, size_t keySize, size_t valCount)
{
  /*
  int i = 0;
  float val_float[2];
  val_float[0] = 0;
  val_float[1] = 0;

  for(i = 0; i < valCount; i++) {
    SAXPY_VAL_T* iVal = (SAXPY_VAL_T*)GET_VAL_FUNC(vals, i);
    val_float[i] = iVal->dst;
  }

  float result = val_float[0] + val_float[1];
  if( result != 0 ) {
  */
    // EMIT_COUNT_FUNC(sizeof(int), sizeof(SAXPY_VAL_T));
    EMIT_COUNT_FUNC(sizeof(int), sizeof(float));
    /*}
     */
}

__device__ void REDUCE_FUNC//(void* key, void* vals, size_t keySize, size_t valCount)
{
#ifdef __NO_REDUCE__
  for(int i = 0; i < valCount; i++)
    // EMIT_FUNC(key, (PR1_VAL_T*)GET_VAL_FUNC(vals,i), sizeof(int), sizeof(SAXPY_VAL_T));
    EMIT_FUNC(key, (float*)GET_VAL_FUNC(vals,i), sizeof(int), sizeof(float));
  return;
#endif 

#ifdef _DEBUG_REDUCE
  printf("reduce: key = %d, valCount = %d\n", *(int*)key, valCount);
#endif
  
  int i = 0;
  float val_float[2];
  val_float[0] = 0;
  val_float[1] = 0;

  if(valCount > 2) 
    printf("error valCount in reduce of Saxpy\n");
  for(i = 0; i < valCount; i++) {
    SAXPY_VAL_T* iVal = (SAXPY_VAL_T*)GET_VAL_FUNC(vals, i);
    val_float[i] = iVal->dst;
  }

  float result = val_float[0] + val_float[1];
  // SAXPY_VAL_T* o_val = (SAXPY_VAL_T*)GET_OUTPUT_BUF(0);
  // float* o_val = (float*)GET_OUTPUT_BUF(0);
  float o_val = result;
  if( result != 0 ) {
    // o_val->is_v = barg;
    // o_val->dst = result;
    // EMIT_FUNC(key, o_val, sizeof(int), sizeof(SAXPY_VAL_T));
    EMIT_FUNC(key, &o_val, sizeof(int), sizeof(float));
  } else {
    int o_key = -1;
    // o_val->is_v = barg;
    // o_val->dst = -1;
    // EMIT_FUNC(&o_key, o_val, sizeof(int), sizeof(SAXPY_VAL_T));
    EMIT_FUNC(&o_key, &o_val, sizeof(int), sizeof(float));
  }
}

#endif //__REDUCE_CU__
