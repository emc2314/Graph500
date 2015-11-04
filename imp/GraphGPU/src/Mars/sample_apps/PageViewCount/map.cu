/*$Id: map.cu 731 2009-11-13 14:45:27Z wenbinor $*/
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

__device__ int hash_func(char* str, int len)
{
	int hash, i;
	for (i = 0, hash=len; i < len; i++)
		hash = (hash<<4)^(hash>>28)^str[i];
	return hash;
}


__device__ void MAP_COUNT_FUNC//(void *key, void *val, size_t keySize, size_t valSize)
{
	PVC_VAL_T *pVal = (PVC_VAL_T*)val;

	if (pVal->phase == 0)
		EMIT_INTER_COUNT_FUNC(sizeof(PVC_KEY_T), sizeof(PVC_VAL_T));
	else
		EMIT_INTER_COUNT_FUNC(sizeof(PVC_KEY_T), sizeof(int));
}

__device__ void MAP_FUNC//(void *key, void val, size_t keySize, size_t valSize)
{
	PVC_KEY_T *pKey = (PVC_KEY_T*)key;
	PVC_VAL_T *pVal = (PVC_VAL_T*)val;

	char* entry = pKey->file_buf + pKey->entry_offset;

	if (pVal->phase == 0)
	{
		pKey->entry_hash = hash_func(entry, pVal->entry_size);
		EMIT_INTERMEDIATE_FUNC(key, val, sizeof(PVC_KEY_T), sizeof(PVC_VAL_T));
	}
	else
	{
		char* p = entry;
		int url_size = 1;
		for (; *p != '\t'; p++, url_size++);
		*p = '\0';
		int ip_offset = pKey->entry_offset + url_size; 
		for (; *p != '\t'; p++);
		*p = '\0';
		//printf("%s | %s\n", pKey->file_buf + pKey->entry_offset, pKey->file_buf + ip_offset);

		int* o_ipoffset = (int*)GET_OUTPUT_BUF(0);
		*o_ipoffset = ip_offset;
		pKey->entry_hash = hash_func(entry, url_size);
		EMIT_INTERMEDIATE_FUNC(key, o_ipoffset, sizeof(PVC_KEY_T), sizeof(int));
	}	
}
#endif //__MAP_CU__
