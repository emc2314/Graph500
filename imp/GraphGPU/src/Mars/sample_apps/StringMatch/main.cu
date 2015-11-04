/*$Id: main.cu 738 2009-11-13 16:08:10Z wenbinor $*/
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
 *String Match (SM): String match is used as exact
 *matching for a string in an input file. Each Map
 *searches one line in the input file to check whether the
 *target string is in the line. For each string it finds, it
 *emits an intermediate pair of the string as the key and
 *the position as the value. No Reduce stage is required.
 ******************************************************************/

#include "MarsInc.h"
#include "global.h"

//#define __OUTPUT__

void printFun(void* key, void* val, int keySize, int valSize)
{
	int* line_offset = (int*)key;
	int* line_size = (int*)val;

	printf("line_offset:%d, line_size:%d\n", *line_offset, *line_size);
}

void validate(Spec_t* spec, int num)
{
	PrintOutputRecords(spec, num, printFun);
}

//-----------------------------------------------------------------------
//usage: StringMatch datafile keyword 
//param: datafile 
//param: keyword
//-----------------------------------------------------------------------
int main( int argc, char** argv) 
{
	if (argc != 3)
	{
		printf("usage: %s datafile keyword\n", argv[0]);
		exit(-1);	
	}
	
	Spec_t *spec = GetDefaultSpec();
	spec->workflow = MAP_ONLY;
#ifdef __OUTPUT__
	spec->outputToHost = 1;
#endif

	TimeVal_t alltimer;
	startTimer(&alltimer);

	TimeVal_t readtimer;
	startTimer(&readtimer);
	char *filename = argv[1];

	FILE *fp = fopen(filename, "r");
	fseek(fp, 0, SEEK_END);
	int fileSize = ftell(fp);
	rewind(fp);
    	char *h_filebuf = (char*)malloc(fileSize);
	char* d_filebuf = NULL;
	fread(h_filebuf, fileSize, 1, fp);
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_filebuf, fileSize));
	CUDA_SAFE_CALL(cudaMemcpy(d_filebuf, h_filebuf, fileSize, cudaMemcpyHostToDevice));
	fclose(fp);

	int keywordSize = strlen(argv[2])+1;
	char* d_keyword = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_keyword, keywordSize));
	CUDA_SAFE_CALL(cudaMemcpy(d_keyword, argv[2], keywordSize, cudaMemcpyHostToDevice));

	SM_KEY_T key;
	key.ptrFile = d_filebuf;
	key.ptrKeyword = d_keyword;
	SM_VAL_T val;
	val.keyword_size = keywordSize;

	int offset = 0;
	char* p = h_filebuf;
	char* start = h_filebuf;
	while (1)
	{
		int blockSize = 1024;
		if (offset + blockSize > fileSize) blockSize = fileSize - offset;
		p += blockSize;
		for (; *p != '\n' && *p != '\0'; p++);	
		if (*p != '\0')
		{
			++p;
			blockSize = (int)(p - start);
			val.linebuf_offset = offset;
			val.linebuf_size = blockSize;
			AddMapInputRecord(spec, &key, &val, sizeof(SM_KEY_T), sizeof(SM_VAL_T));	
			offset += blockSize; 
			start = p;
		}
		else
		{
			*p = '\n';
			blockSize = (int)(fileSize - offset);
			val.linebuf_offset = offset;
			val.linebuf_size = blockSize;
			AddMapInputRecord(spec, &key, &val, sizeof(SM_KEY_T), sizeof(SM_VAL_T));	
			break;
		}
	}
	endTimer("io-test", &readtimer);

	//----------------------------------------------
	//map/reduce
	//----------------------------------------------
	MapReduce(spec);
	endTimer("all-test", &alltimer);

	//----------------------------------------------
	//further processing
	//----------------------------------------------
#ifdef __OUTPUT__
	validate(spec, 10);
#endif
	//----------------------------------------------
	//finish
	//----------------------------------------------
	FinishMapReduce(spec);

	free(h_filebuf);
	CUDA_SAFE_CALL(cudaFree(d_filebuf));
	CUDA_SAFE_CALL(cudaFree(d_keyword));

	return 0;
}
