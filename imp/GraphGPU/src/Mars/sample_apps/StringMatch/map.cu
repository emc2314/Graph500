/*$Id: map.cu 720 2009-11-10 10:13:52Z wenbinor $*/
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
	SM_KEY_T* pKey = (SM_KEY_T*)key;
	SM_VAL_T* pVal = (SM_VAL_T*)val;

	int bufOffset = pVal->linebuf_offset;
	int bufSize = pVal->linebuf_size;
	char* buf = pKey->ptrFile + bufOffset;

	char* keyword =  pKey->ptrKeyword;
	int keywordSize = pVal->keyword_size;

	int cur = 0;
	char* p = buf;
	char* start = buf;

	while(1)
	{
		for (; *p != '\n'; ++p, ++cur);
		++p;
		int wordSize = (int)(p - start);

		if (cur >= bufSize) break;
		char* k = keyword;
		char* s = start;
		if (wordSize == keywordSize) 
		{
			for (; *s == *k && *k != '\0'; s++, k++);
			if (*s == '\n') EMIT_INTER_COUNT_FUNC(sizeof(int), sizeof(int));
		}

		start = p;
		bufOffset += wordSize;
	}
}

__device__ void MAP_FUNC//(void *key, void val, size_t keySize, size_t valSize)
{
	SM_KEY_T* pKey = (SM_KEY_T*)key;
	SM_VAL_T* pVal = (SM_VAL_T*)val;

	int bufOffset = pVal->linebuf_offset;
	int bufSize = pVal->linebuf_size;
	char* buf = pKey->ptrFile + bufOffset;

	char* keyword =  pKey->ptrKeyword;
	int keywordSize = pVal->keyword_size;

	int cur = 0;
	char* p = buf;
	char* start = buf;

	while(1)
	{
		for (; *p != '\n'; ++p, ++cur);
		++p;
		int wordSize = (int)(p - start);
		int wordOffset = bufOffset;

		if (cur >= bufSize) break;

		char* k = keyword;
		char* s = start;

		if (wordSize == keywordSize) 
		{
			for (; *s == *k && *k != '\0'; s++, k++);
			if (*s == '\n') 
			{
				int* o_offset = (int*)GET_OUTPUT_BUF(0);
				int* o_size = (int*)GET_OUTPUT_BUF(sizeof(int));
				*o_offset = wordOffset;
				*o_size = wordSize;
				EMIT_INTERMEDIATE_FUNC(o_offset, o_size, sizeof(int), sizeof(int));
			}
		}

		start = p;
		bufOffset += wordSize;
	}
}
#endif //__MAP_CU__
