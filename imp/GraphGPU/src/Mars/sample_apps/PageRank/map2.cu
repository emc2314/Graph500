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
  EMIT_INTER_COUNT_FUNC(sizeof(PR_KEY_T), sizeof(PR_VAL_T));
}

__device__ void MAP_FUNC2//(void *key, void val, size_t keySize, size_t valSize)
{

//   if(((PR_VAL_T*)val)->is_v == false)
//     printf("XXX %d, %f\n", *(PR_KEY_T*)key, ((PR_VAL_T*)val)->dst);
//   else 
//     printf("XXX %d, v%f\n", *(PR_KEY_T*)key, ((PR_VAL_T*)val)->dst);    

  EMIT_INTERMEDIATE_FUNC(key, val, sizeof(PR_KEY_T), sizeof(PR_VAL_T));
}
#endif //__MAP2_CU__
