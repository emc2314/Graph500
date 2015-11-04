/*$Id: main.cu 737 2009-11-13 15:58:04Z wenbinor $*/
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

/******************************************************************
 * A Template application
 ******************************************************************/

#include "MarsInc.h"
#include "global.h"

void printFun(void* key, void* val, int keySize, int valSize)
{
	int* k = (int*)key;
	int* v = (int*)val;

	printf("(%d, %d)\n", *k, *v);
}

void Output(Spec_t* spec, int num)
{
	if (spec->outputToHost != 1)
	{
		printf("Error: please set outputToHost to 1 first!\n");
		return;
	}
	if (num > spec->outputRecordCount) num = spec->outputRecordCount;

	PrintOutputRecords(spec, num, printFun);	
}

int main( int argc, char** argv) 
{
	Spec_t *spec = GetDefaultSpec();
	//MAP_ONLY, MAP_GROUP, or MAP_REDUCE
	spec->workflow = MAP_REDUCE;
	//1 for outputing result to Host
	//by default, Mars keeps results in device memory
	spec->outputToHost = 1;

	//----------------------------------------------
	//preprocess
	//----------------------------------------------
	TMPL_KEY_T key;
	TMPL_VAL_T val;
	for (int i = 0; i < 10000; i++)
	{
		key.field1 = i/2;
		key.field2 = i/2;	
		val.field1 = i;
		AddMapInputRecord(spec, &key, &val, sizeof(TMPL_KEY_T), sizeof(TMPL_VAL_T));
	}

	//----------------------------------------------
	//start mapreduce
	//----------------------------------------------
	MapReduce(spec);

	//----------------------------------------------
	//further processing
	//----------------------------------------------
	Output(spec, spec->outputRecordCount);

	//----------------------------------------------
	//finish
	//----------------------------------------------
	FinishMapReduce(spec);
	return 0;
}
