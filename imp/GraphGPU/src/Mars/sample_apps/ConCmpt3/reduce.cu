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
//#define _DEBUG_REDUCE

__device__ void REDUCE_COUNT_FUNC//(void* key, void* vals, size_t keySize, size_t valCount)
{
  EMIT_COUNT_FUNC(sizeof(int), sizeof(int));
}

__device__ void REDUCE_FUNC//(void* key, void* vals, size_t keySize, size_t valCount)
{
  int i;
  int sum = 0;
  
  for(i = 0; i < valCount; i++) {
    int* iVal = (int*)GET_VAL_FUNC(vals, i);
    sum += *iVal;
  }
  
  EMIT_FUNC(key, &sum, sizeof(int), sizeof(int));
}

#endif //__REDUCE_CU__
