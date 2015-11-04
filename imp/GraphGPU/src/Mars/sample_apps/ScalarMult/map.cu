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

// #define _DEBUG_MAP

__device__ void MAP_COUNT_FUNC//(void *key, void *val, size_t keySize, size_t valSize)
{
  EMIT_INTER_COUNT_FUNC(sizeof(int), sizeof(float));
}

__device__ void MAP_FUNC//(void *key, void val, size_t keySize, size_t valSize, float farg)
{
  float s = farg;
  float* pVal = (float*)val;
  float o_val = 0;
  o_val = s * (*pVal);
#ifdef _DEBUG_MAP
  printf("map: s = %f, output key = %d, output val = %f\n", s, *(int*)key, o_val);
#endif
  EMIT_INTERMEDIATE_FUNC(key, &o_val, keySize, valSize);
}
#endif //__MAP_CU__
