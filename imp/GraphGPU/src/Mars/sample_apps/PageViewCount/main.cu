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

/***********************************************************************
 *Page View Count (PVC): It obtains the number of
 *distinct page views from the web logs. Each entry in the
 *web log is represented as <URL, IP, Cookie>, where
 *URL is the URL of the accessed page; IP is the IP
 *address that accesses the page; Cookie is the cookie
 *information generated when the page is accessed. This
 *application has two executions of MapReduce. The first
 *one removes the duplicate entries in the web logs. The
 *second one counts the number of page views. In the
 *first MapReduce, each Map takes the pair of an entry as
 *the key and the size of the entry as value. The sort is to
 *eliminate the redundancy in the web log. Specifically, if
 *more than one log entries have the same information,
 *we keep only one of them. The first MapReduce
 *outputs the result pair of the log entry as key and the
 *size of the line as value. The second MapReduce
 *processes the key/value pairs generated from the first
 *MapReduce. The Map outputs the URL as the key and
 *the IP as the value. The Reduce computes the number
 *of IPs for each URL.
 ***********************************************************************/

#include "MarsInc.h"
#include "global.h"

//#define __OUTPUT__

void validate(Spec_t* spec, char* h_filebuf, char* d_filebuf, int fileSize, int num)
{
	if (num > spec->outputDiffKeyCount) num = spec->outputDiffKeyCount;

	CUDA_SAFE_CALL(cudaMemcpy(h_filebuf, d_filebuf, fileSize, cudaMemcpyDeviceToHost));

	for (int i = 0; i < num; i++)
	{
		int2 groupInfo = spec->outputKeyListRange[i];
		PVC_KEY_T* keys = (PVC_KEY_T*)(spec->outputKeys + spec->outputOffsetSizes[groupInfo.x].x);
		int* ip_offsets = (int*)(spec->outputVals + spec->outputOffsetSizes[groupInfo.x].z);

		int groupSize = groupInfo.y - groupInfo.x;
		printf("===========URL: %s - %d unique ip accesses===========\n", h_filebuf + keys->entry_offset, groupSize);
		for (int j = 0; j < groupSize; j++)
		{
			printf("IP: %s\n", h_filebuf + ip_offsets[j]);
		}
	}
}

//-----------------------------------------------------------------
//usage: PageViewCount datafile 
//param: datafile
//-----------------------------------------------------------------
int main( int argc, char** argv) 
{
	if (argc != 2)
	{
		printf("usage: %s datafile\n", argv[0]);
		exit(-1);
	}
	Spec_t *spec = GetDefaultSpec();

	TimeVal_t allTimer;
	startTimer(&allTimer);
	//------------------------------------------------------------------
	//prepare input
	//------------------------------------------------------------------
	TimeVal_t preTimer;
	startTimer(&preTimer);
	FILE *fp = fopen(argv[1], "r");
	fseek(fp, 0, SEEK_END);
	int fileSize = ftell(fp) + 1;
	rewind(fp);
	char *h_filebuf = (char*)malloc(fileSize);
	fread(h_filebuf, fileSize, 1, fp);
	h_filebuf[fileSize-1] = '\n';
	fclose(fp);

	char* d_filebuf = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_filebuf, fileSize));

	char* p = h_filebuf;
	char* start = h_filebuf;
	int cur = 0;
	PVC_KEY_T key;
	key.file_buf = d_filebuf;
	PVC_VAL_T val;
	val.phase = 0;
	while (1)
	{
		for (; *p != '\n'; ++p);
		*p = '\0';
		p++;
		key.entry_offset = cur;
		val.entry_size = p - start;
		cur += val.entry_size;
		if (cur >= fileSize) break;
		AddMapInputRecord(spec, &key, &val, sizeof(PVC_KEY_T), sizeof(PVC_VAL_T));	
		start = p;
	}
	CUDA_SAFE_CALL(cudaMemcpy(d_filebuf, h_filebuf, fileSize, cudaMemcpyHostToDevice));
	endTimer("preprocess", &preTimer);

	//------------------------------------------------------------------
	//the first MapReduce
	//------------------------------------------------------------------
	spec->workflow = MAP_REDUCE;
	MapReduce(spec);

	//------------------------------------------------------------------
	//the second MapReduce
	//------------------------------------------------------------------
	CUDA_SAFE_CALL(cudaMemcpy(spec->inputKeys, spec->outputKeys, spec->outputAllKeySize, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(spec->inputVals, spec->outputVals, spec->outputAllValSize, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(spec->inputOffsetSizes, spec->outputOffsetSizes, spec->outputRecordCount * sizeof(int4), cudaMemcpyDeviceToHost));
	spec->inputRecordCount = spec->outputRecordCount;
	CUDA_SAFE_CALL(cudaFree(spec->outputKeys));
	CUDA_SAFE_CALL(cudaFree(spec->outputVals));
	CUDA_SAFE_CALL(cudaFree(spec->outputOffsetSizes));

	spec->workflow = MAP_GROUP;
#ifdef __OUTPUT__
	spec->outputToHost = 1;
#endif

	MapReduce(spec);

	endTimer("all", &allTimer);
	//------------------------------------------------------------------
	//Further processing
	//------------------------------------------------------------------
#ifdef __OUTPUT__
	validate(spec, h_filebuf, d_filebuf, fileSize, 2);
#endif

	//------------------------------------------------------------------
	//Complete
	//------------------------------------------------------------------
	FinishMapReduce(spec);
	free(h_filebuf);
		
	return 0;
}
