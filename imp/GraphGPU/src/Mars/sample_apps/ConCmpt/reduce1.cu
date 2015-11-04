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

#ifndef __REDUCE1_CU__
#define __REDUCE1_CU__

#include "MarsInc.h"
#include "global.h"

//#define _DEBUG_REDUCE

#define ADD_TO_LIST(list, id, val) do{\
    list[id] = val;\
    id++;\
  }while (0)

__device__ void REDUCE_COUNT_FUNC1//(void* key, void* vals, size_t keySize, size_t valCount)
{
  for(int i = 0; i < valCount; i++) {
    EMIT_COUNT_FUNC(sizeof(int), sizeof(CC_VAL_T));
  }
}

__device__ void REDUCE_FUNC1//(void* key, void* vals, size_t keySize, size_t valCount)
{
  CC_KEY_T* iKey = (CC_KEY_T*)key;

  int i;
  int component_id = -1;
  bool self_contained = false;
  
  int *from_nodes_set = NULL;
  size_t valSize = valCount * sizeof(int); 

  from_nodes_set = (int*)malloc(valSize);
  int from_nodes_set_size = 0;
  
  for(i = 0; i < valCount; i++) {
    CC_VAL_T* iVal = (CC_VAL_T*)GET_VAL_FUNC(vals, i);
    if(iVal->is_v == true) { // component_info
      if(component_id == -1) {
	component_id = iVal->dst;
      }
    }
    else {  // edge line
      int from_node_int = iVal->dst;
      ADD_TO_LIST(from_nodes_set, from_nodes_set_size, from_node_int);
      if(iKey->src == from_node_int)
	self_contained = true;
    }
  }

  if(self_contained == false) // add self loop, if not exists.
    ADD_TO_LIST(from_nodes_set, from_nodes_set_size, iKey->src);

  for(i = 0; i < from_nodes_set_size; i++) {
    CC_VAL_T* o_val = (CC_VAL_T*)GET_OUTPUT_BUF(0);
    int cur_key_int = from_nodes_set[i];
    if(cur_key_int == iKey->src) {  
      o_val->is_v = true; // msi
      o_val->dst = component_id;
      //printf("%d, %d %d\n", cur_key_int, o_val->is_v, o_val->dst);
      EMIT_FUNC(&cur_key_int, o_val, sizeof(int), sizeof(CC_VAL_T));
    } else {
      o_val->is_v = false; // moi
      o_val->dst = component_id;
      EMIT_FUNC(&cur_key_int, o_val, sizeof(int), sizeof(CC_VAL_T));      
    }
  }

  for(i = 0; i < valCount - from_nodes_set_size; i++) {
    CC_VAL_T* o_val = (CC_VAL_T*)GET_OUTPUT_BUF(0);
    int key = -1;
    o_val->is_v = false;
    o_val->dst = -1;
    EMIT_FUNC(&key, o_val, sizeof(int), sizeof(CC_VAL_T));
  }

  free(from_nodes_set);
}

#endif //__REDUCE1_CU__
