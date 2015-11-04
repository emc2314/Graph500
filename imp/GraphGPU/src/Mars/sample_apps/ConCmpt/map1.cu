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

#ifndef __MAP1_CU__
#define __MAP1_CU__

#include "MarsInc.h"
#include "global.h"

//#define _DEBUG_MAP

__device__ void MAP_COUNT_FUNC1//(void *key, void *val, size_t keySize, size_t valSize)
{
    EMIT_INTER_COUNT_FUNC(sizeof(CC_KEY_T), sizeof(CC_VAL_T));
}

__device__ void MAP_FUNC1//(void *key, void val, size_t keySize, size_t valSize)
{
  CC_KEY_T* iKey = (CC_KEY_T*)key;
  CC_VAL_T* iVal = (CC_VAL_T*)val;

#ifdef _DEBUG_MAP
  if(iKey->src == 0)
    printf("map: key = %d, value = %d\n", iKey->src, iVal->dst);
#endif

  if( iVal->is_v == true) { // vector : ROWID  VALUE('vNNNN')
    EMIT_INTERMEDIATE_FUNC(key, val, keySize, valSize);
  } else {
    CC_VAL_T* o_val = (CC_VAL_T*)GET_OUTPUT_BUF(0);
    o_val->is_v = false;
    o_val->dst = iKey->src;
    EMIT_INTERMEDIATE_FUNC(&(iVal->dst), o_val, keySize, valSize);
  }

}
#endif //__MAP1_CU__
