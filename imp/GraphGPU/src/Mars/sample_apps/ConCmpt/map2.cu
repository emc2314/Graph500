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

#ifndef __MAP2_CU__
#define __MAP2_CU__

#include "MarsInc.h"
#include "global.h"

__device__ void MAP_COUNT_FUNC2//(void *key, void *val, size_t keySize, size_t valSize)
{
	if(*(int*)key != -1)
		EMIT_INTER_COUNT_FUNC(sizeof(CC_KEY_T), sizeof(CC_VAL_T));
}

__device__ void MAP_FUNC2//(void *key, void val, size_t keySize, size_t valSize)
{
	if(*(int*)key != -1)
		EMIT_INTERMEDIATE_FUNC(key, val, keySize, valSize);
}
#endif //__MAP2_CU__
