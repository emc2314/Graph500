/*$Id: map.cu 727 2009-11-11 11:32:44Z wenbinor $*/
/**
 *This is the source code for Mars, a MapReduce framework on graphics
 *processors.
 *Developers: Wenbin Fang (HKUST), Bingsheng He (Microsoft Research Asia)
 *Naga K. Govindaraju (Microsoft Corp.), Qiong Luo (HKUST), Tuyong Wang (Sina.com).
 *If you have any question on the code, please contact us at 
 *           wenbin@cse.ust.hk or savenhe@microsoft.com
 *
 *The license is a free non-exclusive, non-transferable license to reproduce, 
 *use, modify and display the source code version of the Software, with or 
 *without modifications solely for non-commercial research, educational or 
 *evaluation purposes. The license does not entitle Licensee to technical support, 
 *telephone assistance, enhancements or updates to the Software. All rights, title 
 *to and ownership interest in Mars, including all intellectual property rights 
 *therein shall remain in HKUST.
 */

#ifndef __MAP_CU__
#define __MAP_CU__

#include "MarsInc.h"
#include "global.h"

__device__ void MAP_COUNT_FUNC//(void *key, void *val, size_t keySize, size_t valSize)
{
	EMIT_INTER_COUNT_FUNC(sizeof(int), sizeof(int));
}

__device__ void MAP_FUNC//(void *key, void val, size_t keySize, size_t valSize)
{
	TMPL_KEY_T* pKey = (TMPL_KEY_T*)key;
	TMPL_VAL_T* pVal = (TMPL_VAL_T*)val;

	int* o_key = (int*)GET_OUTPUT_BUF(0);
	int* o_val = (int*)GET_OUTPUT_BUF(sizeof(int));

	*o_key = pKey->field1;
	*o_val = pVal->field1;

	EMIT_INTERMEDIATE_FUNC(o_key, o_val, sizeof(int), sizeof(int));
}
#endif //__MAP_CU__
