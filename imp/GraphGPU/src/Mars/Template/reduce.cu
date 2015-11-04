/*$Id: reduce.cu 729 2009-11-12 09:56:09Z wenbinor $*/
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

#ifndef __REDUCE_CU__
#define __REDUCE_CU__

#include "MarsInc.h"
#include "global.h"

__device__ void REDUCE_COUNT_FUNC//(void* key, void* vals, size_t keySize, size_t valCount)
{
	EMIT_COUNT_FUNC(sizeof(int), sizeof(int));
}

__device__ void REDUCE_FUNC//(void* key, void* vals, size_t keySize, size_t valCount)
{
	int* pKey = (int*)key;

	int sum = 0;
	for (int i = 0; i < valCount; i++)
	{
		int* val = (int*)GET_VAL_FUNC(vals, i);
		sum += *val;
	}

	int* o_key = (int*)GET_OUTPUT_BUF(0);
	int* o_val = (int*)GET_OUTPUT_BUF(sizeof(int));

	*o_key = sum;
	*o_val = *pKey;
	EMIT_FUNC(o_key, o_val, sizeof(int), sizeof(int));
}
#endif //__REDUCE_CU__
