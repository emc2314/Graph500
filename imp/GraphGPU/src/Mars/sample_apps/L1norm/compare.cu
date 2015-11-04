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

#ifndef __COMPARE_CU__
#define __COMPARE_CU__
#include "MarsInc.h"
#include "global.h"

__device__ int compare(const void *d_a, int len_a, const void *d_b, int len_b)
{

	int key1 = *(int*)d_a;	
	int key2 = *(int*)d_b;	

	if (key1 > key2) return 1;
	if (key1 < key2) return -1;

	return 0;
}

#endif //__COMPARE_CU__
