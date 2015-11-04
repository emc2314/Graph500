/*$Id: compare.cu 731 2009-11-13 14:45:27Z wenbinor $*/
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

#ifndef __COMPARE_CU__
#define __COMPARE_CU__
#include "MarsInc.h"
#include "global.h"

__device__ int compare(const void *d_a, int len_a, const void *d_b, int len_b)
{
	II_KEY_T* a = (II_KEY_T*)d_a;
	II_KEY_T* b = (II_KEY_T*)d_b;

#ifdef __HASH__
	if (a->url_hash > b->url_hash) return 1;
	if (a->url_hash < b->url_hash) return -1;
#endif

	char* url1 = a->file_buf + a->url_offset;
	char* url2 = b->file_buf + b->url_offset;

	for (; *url1 == *url2 && *url1 != '\0' && *url2 != '\0'; url1++, url2++);

	if (*url1 > *url2) return 1;
	if (*url1 < *url2) return -1;
	return 0;
}

#endif //__COMPARE_CU__
