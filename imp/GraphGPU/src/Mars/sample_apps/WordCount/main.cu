/*$Id: main.cu 755 2009-11-18 13:22:54Z wenbinor $*/
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
 *WordCount (WC): It counts the number of occurrences for each word in a file. Each Map
 * task processes a portion of the input file and emits intermediate data pairs, each of which consists
 * of a word as the key and a value of 1 for the occurrence. Group is required, and no reduce is
 * needed, because the Mars runtime provides the size of each group, after the Group stage.
 ******************************************************************/

#include "MarsInc.h"
#include "global.h"
#include <ctype.h>

#define __OUTPUT__

void validate(char* h_filebuf, Spec_t* spec, int num)
{
	char* key = (char*)spec->outputKeys;
	char* val = (char*)spec->outputVals;
	int4* offsetSizes = (int4*)spec->outputOffsetSizes;
	int2* range = (int2*)spec->outputKeyListRange;

	printf("# of words:%d\n", spec->outputDiffKeyCount);
	if (num > spec->outputDiffKeyCount) num = spec->outputDiffKeyCount;
	for (int i = 0; i < num; i++)
	{
		int keyOffset = offsetSizes[range[i].x].x;
		int valOffset = offsetSizes[range[i].x].z;
		char* word = key + keyOffset;
		int wordsize = *(int*)(val + valOffset);
		printf("%s - size: %d - count: %d\n", word, wordsize, range[i].y - range[i].x);
	}
}

//-----------------------------------------------------------------------
//usage: WordCount datafile
//param: datafile 
//-----------------------------------------------------------------------
int main( int argc, char** argv) 
{
	if (argc != 2)
	{
		printf("usage: %s datafile\n", argv[0]);
		exit(-1);	
	}
	
	Spec_t *spec = GetDefaultSpec();
	spec->workflow = MAP_GROUP;
#ifdef __OUTPUT__
	spec->outputToHost = 1;
#endif

	TimeVal_t allTimer;
	startTimer(&allTimer);

	TimeVal_t preTimer;
	startTimer(&preTimer);

	FILE* fp = fopen(argv[1], "r");
	fseek(fp, 0, SEEK_END);
	int fileSize = ftell(fp) + 1;
	rewind(fp);
	char* h_filebuf = (char*)malloc(fileSize);
	char* d_filebuf = NULL;
	fread(h_filebuf, fileSize, 1, fp);
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_filebuf, fileSize));	
	fclose(fp);

	WC_KEY_T key;
	key.file = d_filebuf;

	for (int i = 0; i < fileSize; i++)
		h_filebuf[i] = toupper(h_filebuf[i]);

	WC_VAL_T val;
	int offset = 0;
	char* p = h_filebuf;
	char* start = h_filebuf;
	while (1)
	{
		int blockSize = 2048;
		if (offset + blockSize > fileSize) blockSize = fileSize - offset;
		p += blockSize;
		for (; *p >= 'A' && *p <= 'Z'; p++);
			
		if (*p != '\0') 
		{
			*p = '\0'; 
			++p;
			blockSize = (int)(p - start);
			val.line_offset = offset;
			val.line_size = blockSize;
			AddMapInputRecord(spec, &key, &val, sizeof(WC_KEY_T), sizeof(WC_VAL_T));	
			offset += blockSize;
			start = p;
		}
		else
		{
			*p = '\0'; 
			blockSize = (int)(fileSize - offset);
			val.line_offset = offset;
			val.line_size = blockSize;
			AddMapInputRecord(spec, &key, &val, sizeof(WC_KEY_T), sizeof(WC_VAL_T));	
			break;
		}
	}
	CUDA_SAFE_CALL(cudaMemcpy(d_filebuf, h_filebuf, fileSize, cudaMemcpyHostToDevice));	
	endTimer("preprocess", &preTimer);
	//----------------------------------------------
	//map/reduce
	//----------------------------------------------
	MapReduce(spec);

	endTimer("all", &allTimer);
	//----------------------------------------------
	//further processing
	//----------------------------------------------
#ifdef __OUTPUT__
	CUDA_SAFE_CALL(cudaMemcpy(h_filebuf, d_filebuf, fileSize, cudaMemcpyDeviceToHost));	
	validate(h_filebuf, spec, 10);
#endif
	//----------------------------------------------
	//finish
	//----------------------------------------------
	FinishMapReduce(spec);
	cudaFree(d_filebuf);
	free(h_filebuf);

	return 0;
}
