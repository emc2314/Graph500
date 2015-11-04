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

#ifndef __MAP_CU__
#define __MAP_CU__

#include "MarsInc.h"
#include "global.h"

//#define _DEBUG_MAP

__device__ void MAP_COUNT_FUNC//(void *key, void *val, size_t keySize, size_t valSize)
{
  EMIT_INTER_COUNT_FUNC(sizeof(int), sizeof(SAXPY_VAL_T));
}

__device__ void MAP_FUNC//(void *key, void *val, size_t keySize, size_t valSize)
{

  SAXPY_VAL_T* i_val = (SAXPY_VAL_T*)val;
  if (i_val->is_y == true) { // if ( y_path )
    EMIT_INTERMEDIATE_FUNC(key, val, sizeof(int), sizeof(SAXPY_VAL_T));
  } else { // if ( x_path )
    float a = farg;
    SAXPY_VAL_T* o_val = (SAXPY_VAL_T*)GET_OUTPUT_BUF(0);
    o_val->is_y = false;
    o_val->dst = a * i_val->dst;
    EMIT_INTERMEDIATE_FUNC(key, o_val, sizeof(int), sizeof(SAXPY_VAL_T));
  }

}
#endif //__MAP_CU__
