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

__device__ void MAP_COUNT_FUNC1//(void *key, void *val, size_t keySize, size_t valSize)
{
  EMIT_INTER_COUNT_FUNC(sizeof(RWR_KEY_T), sizeof(RWR_VAL_T));
}

__device__ void MAP_FUNC1//(void *key, void val, size_t keySize, size_t valSize)
{
  //if( val->is_v == true) { // vector : ROWID  VALUE('vNNNN')
    EMIT_INTERMEDIATE_FUNC(key, val, keySize, valSize);
  //} else {
  // In other matrix-vector multiplication, we output (dst, src) here
  // However, In PageRank, the matrix-vector computation formula is M^T * v.
  // Therefore, we output (src,dst) here.
  // EMIT_INTERMEDIATE_FUNC(key, val, keySize, valSize);
  //}

}
#endif //__MAP1_CU__
