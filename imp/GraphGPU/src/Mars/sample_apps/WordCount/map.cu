/*$Id: map.cu 755 2009-11-18 13:22:54Z wenbinor $*/
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
	WC_KEY_T* pKey = (WC_KEY_T*)key;
	WC_VAL_T* pVal = (WC_VAL_T*)val;

	char* ptrBuf = pKey->file + pVal->line_offset;
	int line_size = pVal->line_size;

	char* p = ptrBuf;
	int lsize = 0;
	int wsize = 0;
	char* start = ptrBuf;

	while(1)
	{
		for (; *p >= 'A' && *p <= 'Z'; p++, lsize++);
		*p = '\0';
		++p;
		++lsize;
		wsize = (int)(p - start);
		if (wsize > 6)
		{
			//printf("%s, wsize:%d\n", start, wsize);	
			EMIT_INTER_COUNT_FUNC(wsize, sizeof(int));
		}
		for (; (lsize < line_size) && (*p < 'A' || *p > 'Z'); p++, lsize++);
		if (lsize >= line_size) break;
		start = p;
	}
}

__device__ void MAP_FUNC//(void *key, void val, size_t keySize, size_t valSize)
{
	WC_KEY_T* pKey = (WC_KEY_T*)key;
	WC_VAL_T* pVal = (WC_VAL_T*)val;

	char* filebuf = pKey->file;
	char* ptrBuf = filebuf + pVal->line_offset;
	int line_size = pVal->line_size;

	char* p = ptrBuf;
	char* start = ptrBuf;
	int lsize = 0;
	int wsize = 0;

	while(1)
	{
		for (; *p >= 'A' && *p <= 'Z'; p++, lsize++);
		*p = '\0';
		++p;
		++lsize;
		wsize = (int)(p - start);
		int* o_val = (int*)GET_OUTPUT_BUF(0);
		*o_val = wsize;
		if (wsize > 6) 
		{
			//printf("%s, %d\n", start, wsize);	
			EMIT_INTERMEDIATE_FUNC(start, o_val, wsize, sizeof(int));
		}
		for (; (lsize < line_size) && (*p < 'A' || *p > 'Z'); p++, lsize++);
		if (lsize >= line_size) break;
		start = p;	
	}
}
#endif //__MAP_CU__
