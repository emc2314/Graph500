/*$Id: MarsUtils.cpp 721 2009-11-10 10:23:55Z wenbinor $*/
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

#ifndef __MRUTILS_CU__
#define __MRUTILS_CU__

#include "MarsInc.h"

//--------------------------------------------------------
//start a timer
//
//param	: start_tv
//--------------------------------------------------------
void startTimer(TimeVal_t *start_tv)
{
   gettimeofday((struct timeval*)start_tv, NULL);
}

//--------------------------------------------------------
//end a timer, and print out a message
//
//param	: msg message to print out
//param	: start_tv
//--------------------------------------------------------
void endTimer(char *msg, TimeVal_t *start_tv)
{
	cudaThreadSynchronize();
   struct timeval end_tv;

   gettimeofday(&end_tv, NULL);

   time_t sec = end_tv.tv_sec - start_tv->tv_sec;
   time_t ms = end_tv.tv_usec - start_tv->tv_usec;

   time_t diff = sec * 1000000 + ms;

   printf("%10s:\t\t%fms\n", msg, (double)((double)diff/1000.0));
}


//----------------------------------------------------------
//print output records
//
//param: spec
//param: num -- maximum number of output records to print
//param: printFunc -- a function pointer
//	void printFunc(void* key, void* val, int keySize, int valSize)
//----------------------------------------------------------
void PrintOutputRecords(Spec_t* spec, int num, PrintFunc_t printFunc)
{
	int maxNum = num;
	if (maxNum > spec->outputRecordCount || maxNum < 0) maxNum = spec->outputRecordCount;
	for (int i = 0; i < maxNum; ++i)
	{
		int4 index = spec->outputOffsetSizes[i];
		printFunc((char*)spec->outputKeys + index.x, (char*)spec->outputVals + index.z, index.y, index.w);
	}
}

#endif //__MRUTILS_CU__
