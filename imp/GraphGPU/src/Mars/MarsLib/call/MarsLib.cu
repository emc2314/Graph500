/*$Id: MarsLib.cu 756 2009-11-18 13:23:58Z wenbinor $*/
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

#ifndef __MRLIB_CU__
#define __MRLIB_CU__

#include "MarsInc.h"
#include "map.cu"
#include "reduce.cu"

//----------------------------------------------
//Get default runtime configuration
//
//return: default spec
//----------------------------------------------
Spec_t *GetDefaultSpec()
{
	Spec_t *spec = (Spec_t*)malloc(sizeof(Spec_t));
	if (NULL == spec) exit(-1);
	memset(spec, 0, sizeof(Spec_t));

	return spec;
}

//--------------------------------------------------------
//Initiate map reduce spec
//--------------------------------------------------------
void InitMapReduce(Spec_t* spec)
{
	Spec_t* g_spec = spec;
	if (g_spec->dimBlockMap <= 0)
		g_spec->dimBlockMap = DEFAULT_DIMBLOCK;

	if (g_spec->dimBlockReduce <= 0)
		g_spec->dimBlockReduce = DEFAULT_DIMBLOCK;

	if (g_spec->numRecTaskReduce <= 0)
		g_spec->numRecTaskReduce = DEFAULT_NUMTASK;

	if (g_spec->numRecTaskMap <= 0)
		g_spec->numRecTaskMap = DEFAULT_NUMTASK;

	if (g_spec->workflow <= 0)
		g_spec->workflow = MAP_ONLY;
}

//--------------------------------------------------
//Add a map input record
//
//param	: spec
//param	: key -- a pointer to a buffer
//param	: val -- a pointer to a buffer
//param	: keySize
//param	: valSize
//--------------------------------------------------
void AddMapInputRecord(Spec_t*		spec, 
		   void*		key, 
		   void*		val,
		   int		keySize, 
		   int		valSize)
{
	assert(NULL != spec);

	static int2 curOffset;
	static int3 curChunkNum;

	int index = spec->inputRecordCount;

	const int dataChunkSize = 1024*1024*256;

	if (spec->inputRecordCount > 0)
	{
		if (dataChunkSize*curChunkNum.x < (curOffset.x + keySize))
			spec->inputKeys = (char*)realloc(spec->inputKeys, (++curChunkNum.x)*dataChunkSize);
		memcpy(spec->inputKeys+curOffset.x, key, keySize);

		if (dataChunkSize*curChunkNum.y < (curOffset.y + valSize))
			spec->inputVals = (char*)realloc(spec->inputVals, (++curChunkNum.y)*dataChunkSize);
		memcpy(spec->inputVals+curOffset.y, val, valSize);		

		if (dataChunkSize*curChunkNum.z < (spec->inputRecordCount+1)*sizeof(int4))
			spec->inputOffsetSizes = (int4*)realloc(spec->inputOffsetSizes, 
			(++curChunkNum.z)*dataChunkSize);
	}
	else
	{
		spec->inputKeys = (char*)malloc(dataChunkSize);
		if (NULL == spec->inputKeys) exit(-1);
		memcpy(spec->inputKeys, key, keySize);

		spec->inputVals = (char*)malloc(dataChunkSize);

		if (NULL == spec->inputVals) exit(-1);
		memcpy(spec->inputVals, val, valSize);

		spec->inputOffsetSizes = (int4*)malloc(dataChunkSize);

		curChunkNum.x++;
		curChunkNum.y++;
		curChunkNum.z++;
	}

	spec->inputOffsetSizes[index].x = curOffset.x;
	spec->inputOffsetSizes[index].y = keySize;
	spec->inputOffsetSizes[index].z = curOffset.y;
	spec->inputOffsetSizes[index].w = valSize;

	curOffset.x += keySize;
	curOffset.y += valSize;

	spec->inputRecordCount++;
}


//-------------------------------------------------
//Called by user defined map_count function
//
//param	: keySize
//param	: valSize
//param	: interKeysSizePerTask
//param	: interValsSizePerTask
//param	: interCountPerTask
//-------------------------------------------------
__device__ void EmitInterCount(int	keySize,
						       int	valSize,
						       int*	interKeysSizePerTask,
						       int*	interValsSizePerTask,
						       int*	interCountPerTask)
{
	int index = TID;

	interKeysSizePerTask[index] += keySize;
	interValsSizePerTask[index] += valSize;
	interCountPerTask[index]++;
}


//-------------------------------------------------
//called by user defined map function
//
//-------------------------------------------------
__device__ void EmitIntermediate(void*		key, 
				 void*		val, 
				 int		keySize, 
				 int		valSize,
				 int*	psKeySizes,
				 int*	psValSizes,
				 int*	psCounts,
				 int2*		keyValOffsets,
				 char*		interKeys,
				 char*		interVals,
				 int4*		interOffsetSizes,
				 int*	curIndex)
{
#ifndef __DEVICE_EMULATION__
	__syncthreads();
#endif
	int index = TID;

	int2 l_keyValOffsets = keyValOffsets[index];

	char *pKeySet = (char*)(interKeys + psKeySizes[index] + l_keyValOffsets.x);
	char *pValSet = (char*)(interVals + psValSizes[index] + l_keyValOffsets.y);
   
	char* sKey = (char*)key;
	char* sVal = (char*)val;
	for (int i = 0; i < keySize; ++i)
		pKeySet[i] = sKey[i];
	for (int i = 0; i < valSize; ++i)
		pValSet[i] = sVal[i];

	l_keyValOffsets.x += keySize;
	l_keyValOffsets.y += valSize;

	keyValOffsets[index] = l_keyValOffsets;

	int l_curIndex = curIndex[index];
	int l_psCounts = psCounts[index];
	int l_curPs = l_curIndex + l_psCounts;
	int4 l_interOffsetSizes1 = interOffsetSizes[l_curPs];
	int4 l_interOffsetSizes2 = interOffsetSizes[l_curPs-1];

	if (l_curIndex != 0)
	{
	     l_interOffsetSizes1.x = (l_interOffsetSizes2.x + l_interOffsetSizes2.y);
	     l_interOffsetSizes1.z = (l_interOffsetSizes2.z + l_interOffsetSizes2.w);
	}
	
	l_interOffsetSizes1.y = keySize;
	l_interOffsetSizes1.w = valSize;
	interOffsetSizes[l_curPs] = l_interOffsetSizes1;

	++l_curIndex;
	curIndex[index] = l_curIndex;
}

//-------------------------------------------------
//Calculate intermediate data's size
//
//param	: inputKeys
//param	: inputVals
//param	: inputOffsetSizes
//param	: interKeysSizesPerTask
//param	: interValsSizePerTask
//param	: interCountPerTask
//param	: recordNum	-- total number of records
//param	: recordsPerTask
//-------------------------------------------------
__global__ void MapperCount(char*	inputKeys,
			char*	inputVals,
			int4*	inputOffsetSizes,
			int*	interKeysSizePerTask,
			int*	interValsSizePerTask,
			int*	interCountPerTask,
			int recordNum, 
			int recordsPerTask,
			int taskNum)
{
	int index = TID;
	int bid = BLOCK_ID;
	int tid = THREAD_ID;
	if (index*recordsPerTask >= recordNum) return;
	int recordBase = bid * recordsPerTask * blockDim.x;
	int terminate = (bid + 1) * (recordsPerTask * blockDim.x);
	if (terminate > recordNum) terminate = recordNum;

	for (int i = recordBase + tid; i < terminate; i+=blockDim.x)
	{
		int cindex = i;
		int4 offsetSize = inputOffsetSizes[cindex];
		char *key = inputKeys + offsetSize.x;
		char *val = inputVals + offsetSize.z;
		map_count(key,
		          val,
			  offsetSize.y,
			  offsetSize.w,
			  interKeysSizePerTask,
			  interValsSizePerTask,
			  interCountPerTask);
	}
}

//--------------------------------------------------
//mapper
//--------------------------------------------------
__global__ void Mapper(char*	inputKeys,
		   char*	inputVals,
		   int4*	inputOffsetSizes,
		   int*	psKeySizes,
		   int*	psValSizes,
		   int*	psCounts,
		   int2*	keyValOffsets,
		   char*	interKeys,
		   char*	interVals,
		   int4*	interOffsetSizes,
		   int*	curIndex,
		   int	recordNum, 
		   int	recordsPerTask,
		   int	taskNum)
{
	int index = TID;
	int bid = BLOCK_ID;
	int tid = THREAD_ID;

	if (index*recordsPerTask >= recordNum) return;
	int recordBase = bid * recordsPerTask * blockDim.x;
	int terminate = (bid + 1) * (recordsPerTask * blockDim.x);
	if (terminate > recordNum) terminate = recordNum;

	int l_psCounts = psCounts[index];
	int4 l_interOffsetSizes = interOffsetSizes[l_psCounts];
	l_interOffsetSizes.x = psKeySizes[index];
	l_interOffsetSizes.z = psValSizes[index];
	interOffsetSizes[l_psCounts] = l_interOffsetSizes;

	for (int i = recordBase + tid; i < terminate; i+=blockDim.x)
	{
		int cindex =  i;

		int4 offsetSize = inputOffsetSizes[cindex];
		char *key = inputKeys + offsetSize.x;
		char *val = inputVals + offsetSize.z;
	
		map(key,
		val,
		offsetSize.y,
		offsetSize.w,
		psKeySizes,
		psValSizes,
		psCounts,
		keyValOffsets,
		interKeys,
		interVals,
		interOffsetSizes,
		curIndex);
	}	
}

//--------------------------------------------------
//start map
//
//1, get map input data on host
//2, upload map input data to device memory
//	 (keys, vals, keyOffsets, valOffsets, keySizes, valSizes)
//3, determine the number of threads to run 
//4, calculate intermediate data keys'buf size 
//	 and values' buf size
//5, do prefix sum on--
//	 i)		d_interKeysSizePerTask
//	 ii)	d_interValsSizePerTask
//	 iii)	d_interCountPerTask
//6, allocate intermediate memory on device memory
//7, start map
//8, free allocated memory
//--------------------------------------------------
int startMap(Spec_t* spec)
{
	Spec_t* g_spec = spec;

	if (g_spec->inputKeys == NULL) { DoLog("Error: no any input keys"); exit(0);}
	if (g_spec->inputVals == NULL) { DoLog("Error: no any input values"); exit(0); }
	if (g_spec->inputOffsetSizes == NULL) { DoLog( "Error: no any input pointer info"); exit(0); }
	if (g_spec->inputRecordCount == 0) {DoLog( "Error: invalid input record count"); exit(0);}
	
	//-------------------------------------------------------
	//1, get map input data on host
	//-------------------------------------------------------
	int	h_inputRecordCount = g_spec->inputRecordCount;
	int	h_inputKeysBufSize = g_spec->inputOffsetSizes[h_inputRecordCount-1].x +
								 g_spec->inputOffsetSizes[h_inputRecordCount-1].y;
	int	h_inputValsBufSize = g_spec->inputOffsetSizes[h_inputRecordCount-1].z +
								 g_spec->inputOffsetSizes[h_inputRecordCount-1].w;
	char*	h_inputKeys = g_spec->inputKeys;
	char*	h_inputVals = g_spec->inputVals;
	int4*	h_inputOffsetSizes = g_spec->inputOffsetSizes;
	DoLog( "** Map Input: keys buf size %d bytes, vals buf size %d bytes, index buf size %d bytes, %d records",
		h_inputKeysBufSize, h_inputValsBufSize, sizeof(int4)*h_inputRecordCount, h_inputRecordCount);	

	//-------------------------------------------------------
	//2, upload map input data onto device memory
	//-------------------------------------------------------
	DoLog( "** Upload map input data onto device memory");
	TimeVal_t uploadTv;
	startTimer(&uploadTv);
	char*	d_inputKeys = NULL;
	char*	d_inputVals = NULL;
	int4*	d_inputOffsetSizes = NULL;

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_inputKeys, h_inputKeysBufSize));
	CUDA_SAFE_CALL(cudaMemcpy(d_inputKeys, h_inputKeys, h_inputKeysBufSize, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_inputVals, h_inputValsBufSize));
	CUDA_SAFE_CALL(cudaMemcpy(d_inputVals, h_inputVals, h_inputValsBufSize, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_inputOffsetSizes, sizeof(int4)*h_inputRecordCount));
	cudaMemcpy(d_inputOffsetSizes, h_inputOffsetSizes, sizeof(int4)*h_inputRecordCount, cudaMemcpyHostToDevice);
	endTimer("PCI-E I/O", &uploadTv);
 
	//----------------------------------------------
	//3, determine the number of threads to run
	//----------------------------------------------
	dim3 h_dimBlock(g_spec->dimBlockMap,1,1);
	dim3 h_dimGrid(1,1,1);
	int h_recordsPerTask = g_spec->numRecTaskMap;
	int numBlocks = CEIL(CEIL(h_inputRecordCount, h_recordsPerTask), h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);
	int h_actualNumThreads = h_dimGrid.x*h_dimBlock.x*h_dimGrid.y;
 
	TimeVal_t mapTimer;
	startTimer(&mapTimer);
	//----------------------------------------------
	//4, calculate intermediate data keys'buf size 
	//	 and values' buf size
	//----------------------------------------------
	DoLog( "** MapCount");
	int*	d_interKeysSizePerTask = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_interKeysSizePerTask, sizeof(int)*h_actualNumThreads));
	cudaMemset(d_interKeysSizePerTask, 0, sizeof(int)*h_actualNumThreads);

	int*	d_interValsSizePerTask = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_interValsSizePerTask, sizeof(int)*h_actualNumThreads));
	cudaMemset(d_interValsSizePerTask, 0, sizeof(int)*h_actualNumThreads);

	int*	d_interCountPerTask = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_interCountPerTask, sizeof(int)*h_actualNumThreads));
	cudaMemset(d_interCountPerTask, 0, sizeof(int)*h_actualNumThreads);
	MapperCount<<<h_dimGrid, h_dimBlock>>>(d_inputKeys,
                                               d_inputVals,
					       d_inputOffsetSizes,
					       d_interKeysSizePerTask,
					       d_interValsSizePerTask,
					       d_interCountPerTask,
					       h_inputRecordCount, 
					       h_recordsPerTask,
					       h_actualNumThreads);
 
 	cudaThreadSynchronize(); 
	//-----------------------------------------------
	//5, do prefix sum on--
	//	 i)		d_interKeysSizePerTask
	//	 ii)	d_interValsSizePerTask
	//	 iii)	d_interCountPerTask
	//-----------------------------------------------
	DoLog( "** Do prefix sum on intermediate data's size\n");
	int *d_psKeySizes = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_psKeySizes, sizeof(int)*h_actualNumThreads));
	int h_allKeySize = prefexSum((int*)d_interKeysSizePerTask, (int*)d_psKeySizes, h_actualNumThreads);

	int *d_psValSizes = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_psValSizes, sizeof(int)*h_actualNumThreads));
	int h_allValSize = prefexSum((int*)d_interValsSizePerTask, (int*)d_psValSizes, h_actualNumThreads);

	int *d_psCounts = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_psCounts, sizeof(int)*h_actualNumThreads));
	int h_allCounts = prefexSum((int*)d_interCountPerTask, (int*)d_psCounts, h_actualNumThreads);

	DoLog( "** Map Output: keys buf size %d bytes, vals buf size %d bytes, index buf size %d bytes, %d records", 
		h_allKeySize, h_allValSize, h_allCounts * sizeof(int4), h_allCounts);


	if (h_allCounts == 0)
	{
		DoLog( "** No output.");

		cudaFree(d_inputKeys);
		cudaFree(d_inputVals);
		cudaFree(d_inputOffsetSizes);
	
		cudaFree(d_psKeySizes);
		cudaFree(d_psValSizes);
		cudaFree(d_psCounts);

		endTimer("Map", &mapTimer);
		return 1;
	}

	//-----------------------------------------------
	//6, allocate intermediate memory on device memory
	//-----------------------------------------------
	DoLog( "** Allocate intermediate memory on device memory");
	char*	d_interKeys = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_interKeys, h_allKeySize));
	cudaMemset(d_interKeys, 0, h_allKeySize);
	
	char*	d_interVals = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_interVals, h_allValSize));
	cudaMemset(d_interVals, 0, h_allValSize);

	int4*	d_interOffsetSizes = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_interOffsetSizes, sizeof(int4)*h_allCounts));
	cudaMemset(d_interOffsetSizes, 0, sizeof(int4)*h_allCounts);

	//--------------------------------------------------
	//7, start map
	//--------------------------------------------------
	DoLog( "** Map");
	
	int2*	d_keyValOffsets = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_keyValOffsets, sizeof(int2)*h_actualNumThreads));
	cudaMemset(d_keyValOffsets, 0, sizeof(int2)*h_actualNumThreads);

	int*	d_curIndex = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_curIndex, sizeof(int)*h_actualNumThreads));
	cudaMemset(d_curIndex, 0, sizeof(int)*h_actualNumThreads);

	int sizeSmem = h_dimBlock.x * sizeof(int) * 5;
	Mapper<<<h_dimGrid, h_dimBlock, sizeSmem>>>(d_inputKeys,
					  d_inputVals,
					  d_inputOffsetSizes,
					  d_psKeySizes,
					  d_psValSizes,
					  d_psCounts,
					  d_keyValOffsets,
					  d_interKeys,
					  d_interVals,
					  d_interOffsetSizes,
					  d_curIndex,
					  h_inputRecordCount, 
					  h_recordsPerTask,
					  h_actualNumThreads);
 	cudaThreadSynchronize(); 
	g_spec->interKeys = d_interKeys;
	g_spec->interVals = d_interVals;
	g_spec->interOffsetSizes = d_interOffsetSizes;
	g_spec->interRecordCount = h_allCounts;
	g_spec->interDiffKeyCount = h_allCounts;
	g_spec->interAllKeySize = h_allKeySize;
	g_spec->interAllValSize = h_allValSize;

	//----------------------------------------------
	//8, free
	//----------------------------------------------

	cudaFree(d_interKeysSizePerTask);
	cudaFree(d_interValsSizePerTask);
	cudaFree(d_interCountPerTask);
	
	cudaFree(d_keyValOffsets);
	cudaFree(d_curIndex);

	cudaFree(d_inputKeys);
	cudaFree(d_inputVals);
	cudaFree(d_inputOffsetSizes);
	
	cudaFree(d_psKeySizes);
	cudaFree(d_psValSizes);
	cudaFree(d_psCounts);

	endTimer("Map", &mapTimer);
	return 0;
}

void startGroup(Spec_t* spec)
{
	Spec_t* g_spec = spec;

	int 	interDiffKeyCount = 0;
	char*	d_outputKeys = NULL;
	char*	d_outputVals = NULL;
	int4*	d_outputOffsetSizes = NULL;
	int2**	h_outputKeyListRange = NULL;

	DoLog( "** Sort for group");

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_outputKeys, g_spec->interAllKeySize));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_outputVals, g_spec->interAllValSize));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_outputOffsetSizes, sizeof(int4)*g_spec->interRecordCount));

	h_outputKeyListRange = (int2**)malloc(sizeof(int2*));

	saven_initialPrefixSum(g_spec->interRecordCount);

	interDiffKeyCount = 
		sort_GPU (g_spec->interKeys, 
				  g_spec->interAllKeySize, 
				  g_spec->interVals, 
				  g_spec->interAllValSize, 
				  g_spec->interOffsetSizes, 
				  g_spec->interRecordCount, 
				  d_outputKeys, 
				  d_outputVals, 
				  d_outputOffsetSizes, 
				  h_outputKeyListRange);
	DoLog( "** InterRecordCount:%d, number of groups: %d", g_spec->interRecordCount, interDiffKeyCount);
	g_spec->interKeys = d_outputKeys;
	g_spec->interVals = d_outputVals;
	g_spec->interOffsetSizes = d_outputOffsetSizes;
	g_spec->interDiffKeyCount = interDiffKeyCount;
	int keyListRangeSize = g_spec->interDiffKeyCount * sizeof(int2);
	CUDA_SAFE_CALL(cudaMalloc((void**)&g_spec->interKeyListRange, keyListRangeSize));
	CUDA_SAFE_CALL(cudaMemcpy(g_spec->interKeyListRange, *h_outputKeyListRange, keyListRangeSize, cudaMemcpyHostToDevice));
	free(*h_outputKeyListRange);
	free(h_outputKeyListRange);
}



//--------------------------------------------------------
//get a value from value list of the same key
//
//param	: vals
//param	: interOffsetSizes
//param	: index
//return: the wanted value
//--------------------------------------------------------
__device__ void *GetVal(void *vals, int4* interOffsetSizes, int keyIndex, int valStartIndex)
{
	int4 offset = interOffsetSizes[valStartIndex];
	return (void*)((char*)vals + keyIndex * offset.w);

}

__device__ void *GetKey(void *key, int4* interOffsetSizes, int keyIndex, int valStartIndex)
{
	int4 offset = interOffsetSizes[valStartIndex];
	return (void*)((char*)key + keyIndex * offset.y);
}

//---------------------------------------------------------
//called by user defined reduce_count function
//---------------------------------------------------------
__device__ void EmitCount(int		keySize,
						  int		valSize,
						  int*		outputKeysSizePerTask,
						  int*		outputValsSizePerTask,
						  int*		outputCountPerTask)
{
	int index = TID;

	outputKeysSizePerTask[index] += keySize;
	outputValsSizePerTask[index] += valSize;
	outputCountPerTask[index]++;
}

//---------------------------------------------------------
//called by user defined reduce function
//---------------------------------------------------------
__device__ void Emit  (char*		key, 
					   char*		val, 
					   int		keySize, 
					   int		valSize,
					   int*		psKeySizes, 
					   int*		psValSizes, 
					   int*		psCounts, 
					   int2*		keyValOffsets, 
					   char*		outputKeys,
					   char*		outputVals,
					   int4*		outputOffsetSizes,
					   int*		curIndex)
{
#ifndef __DEVICE_EMULATION__
	__syncthreads();
#endif
	int index = TID;

	char *pKeySet = (char*)(outputKeys + psKeySizes[index] + keyValOffsets[index].x);
	char *pValSet = (char*)(outputVals + psValSizes[index] + keyValOffsets[index].y);

	for (int i = 0; i < keySize; i++)
		pKeySet[i] = key[i];
	for (int i = 0; i < valSize; i++)
		pValSet[i] = val[i];

	keyValOffsets[index].x += keySize;
	keyValOffsets[index].y += valSize;

	if (curIndex[index] != 0)
	{
	outputOffsetSizes[psCounts[index] + curIndex[index]].x = 
		(outputOffsetSizes[psCounts[index] + curIndex[index] - 1].x + 
		 outputOffsetSizes[psCounts[index] + curIndex[index] - 1].y);
	outputOffsetSizes[psCounts[index] + curIndex[index]].z = 
		(outputOffsetSizes[psCounts[index] + curIndex[index] - 1].z + 
		 outputOffsetSizes[psCounts[index] + curIndex[index] - 1].w);
	}
	
	outputOffsetSizes[psCounts[index] + curIndex[index]].y = keySize;
	outputOffsetSizes[psCounts[index] + curIndex[index]].w = valSize;
	curIndex[index]++;
}

//-------------------------------------------------------
//calculate output data's size
//-------------------------------------------------------
__global__	void ReducerCount(char*		interKeys,
							  char*	    interVals,
							  int4*	    interOffsetSizes,
							  int2*		interKeyListRange,
							  int*   outputKeysSizePerTask,
							  int*   outputValsSizePerTask,
							  int*   outputCountPerTask,
							  int    recordNum, 
							  int    recordsPerTask,
							int	taskNum)
{
	int index = TID;
	int bid = BLOCK_ID;
	int tid = THREAD_ID;

	if (index*recordsPerTask >= recordNum) return;
	int recordBase = bid * recordsPerTask * blockDim.x;
	int terminate = (bid + 1) * (recordsPerTask * blockDim.x);
	if (terminate > recordNum) terminate = recordNum;

	//for (int i = 0; i <= recordsPerTask; i++)
	for (int i = recordBase + tid; i < terminate; i+=blockDim.x)
	{
		int cindex = i;
	
		int valStartIndex = interKeyListRange[cindex].x;
		int valCount = interKeyListRange[cindex].y - interKeyListRange[cindex].x;

		int keySize = interOffsetSizes[interKeyListRange[cindex].x].y;

		char *key = interKeys + interOffsetSizes[valStartIndex].x;
		char *vals = interVals + interOffsetSizes[valStartIndex].z;

		reduce_count(key,
		             vals,
				     keySize,
				 valCount,
					 interOffsetSizes,
				     outputKeysSizePerTask,
				     outputValsSizePerTask,
				     outputCountPerTask);
	}
}

//-------------------------------------------------------
//Reducer
//
//-------------------------------------------------------
__global__ void Reducer(char*		interKeys,
						char*		interVals,
						int4*		interOffsetSizes,
						int2*		interKeyListRange,
					    int*		psKeySizes,
					    int*		psValSizes,
					    int*		psCounts,
						char*		outputKeys,
						char*		outputVals,
						int4*		outputOffsetSizes,
						int2*		keyValOffsets,
						int*		curIndex,
						int		recordNum, 
						int		recordsPerTask,
						int		taskNum)
{
	int index = TID;
	int bid = BLOCK_ID;
	int tid = THREAD_ID;

	if (index*recordsPerTask >= recordNum) return;
	int recordBase = bid * recordsPerTask * blockDim.x;
	int terminate = (bid + 1) * (recordsPerTask * blockDim.x);
	if (terminate > recordNum) terminate = recordNum;


	outputOffsetSizes[psCounts[index]].x = psKeySizes[index];
	outputOffsetSizes[psCounts[index]].z = psValSizes[index];

	for (int i = recordBase + tid; i < terminate; i+=blockDim.x)
	{
		int cindex = i;
	
		int valStartIndex = interKeyListRange[cindex].x;
		int valCount = interKeyListRange[cindex].y - interKeyListRange[cindex].x;

		int keySize = interOffsetSizes[interKeyListRange[cindex].x].y;

		char *key = interKeys + interOffsetSizes[valStartIndex].x;
		char *vals = interVals + interOffsetSizes[valStartIndex].z;

		reduce(key,
			   vals,
			   keySize,
			   valCount,
			   psKeySizes,
			   psValSizes,
			   psCounts,
			   keyValOffsets,
			   interOffsetSizes,
			   outputKeys,
			   outputVals,
			   outputOffsetSizes,
			   curIndex,
			valStartIndex);
	}
}

//----------------------------------------------
//start reduce
//
//1, if there is not a reduce phase, just return
//   then user uses spec->interKeys/spec->intervals 
//   for further processing
//2, get reduce input data on host
//3, upload reduce input data onto device memory
//4, determine the number of threads to run
//5, calculate output data keys'buf size 
//	 and values' buf size
//6, do prefix sum on--
//	 i)		d_outputKeysSizePerTask
//	 ii)	d_outputValsSizePerTask
//	 iii)	d_outputCountPerTask
//7, allocate output memory on device memory
//8, start reduce
//9, copy output data to Spect_t structure
//10,free allocated memory
//----------------------------------------------
void startReduce(Spec_t* spec)
{
	Spec_t* g_spec = spec;

	if (g_spec->interKeys == NULL) {DoLog( "Error: no any intermediate keys"); exit(0);}
	if (g_spec->interVals == NULL) {DoLog( "Error: no any intermediate values"); exit(0);}
	if (g_spec->interOffsetSizes == NULL) {DoLog( "Error: no any intermediate pointer info");exit(0);}
	if (g_spec->interRecordCount == 0) {DoLog( "Error: invalid intermediate record count");exit(0);}
	if (g_spec->interKeyListRange == NULL) { DoLog( "Error: no any key list range");exit(0);}
	if (g_spec->interDiffKeyCount == 0) { DoLog( "Error: invalid intermediate diff key count");exit(0);}
	
	//-------------------------------------------------------
	//2, get reduce input data on host
	//-------------------------------------------------------
	int	h_interDiffKeyCount = g_spec->interDiffKeyCount;
	char*	d_interKeys = g_spec->interKeys;
	char*	d_interVals = g_spec->interVals;
	int4*	d_interOffsetSizes = g_spec->interOffsetSizes;
	int2* 	d_interKeyListRange = g_spec->interKeyListRange;

	//----------------------------------------------
	//4, determine the number of threads to run
	//----------------------------------------------
	dim3 h_dimBlock(g_spec->dimBlockReduce,1,1);
	dim3 h_dimGrid(1,1,1);
	int h_recordsPerTask = g_spec->numRecTaskReduce;
	int numBlocks = CEIL(CEIL(h_interDiffKeyCount, h_recordsPerTask), h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);
	int h_actualNumThreads = h_dimGrid.x*h_dimBlock.x*h_dimGrid.y;

	//----------------------------------------------
	//5, calculate output data keys'buf size 
	//	 and values' buf size
	//----------------------------------------------
	DoLog( "** ReduceCount");
	int*	d_outputKeysSizePerTask = NULL;
	cudaMalloc((void**)&d_outputKeysSizePerTask, sizeof(int)*h_actualNumThreads);
	cudaMemset(d_outputKeysSizePerTask, 0, sizeof(int)*h_actualNumThreads);

	int*	d_outputValsSizePerTask = NULL;
	cudaMalloc((void**)&d_outputValsSizePerTask, sizeof(int)*h_actualNumThreads);
	cudaMemset(d_outputValsSizePerTask, 0, sizeof(int)*h_actualNumThreads);

	int*	d_outputCountPerTask = NULL;
	cudaMalloc((void**)&d_outputCountPerTask, sizeof(int)*h_actualNumThreads);
	cudaMemset(d_outputCountPerTask, 0, sizeof(int)*h_actualNumThreads);

	ReducerCount<<<h_dimGrid, h_dimBlock>>>(d_interKeys,
					    d_interVals,
					    d_interOffsetSizes,
					d_interKeyListRange,
				    d_outputKeysSizePerTask,
			    d_outputValsSizePerTask,
			    d_outputCountPerTask,
			    h_interDiffKeyCount, 
		    h_recordsPerTask,
			h_actualNumThreads);
	cudaThreadSynchronize();
	//-----------------------------------------------
	//6, do prefix sum on--
	//	 i)		d_outputKeysSizePerTask
	//	 ii)	d_outputValsSizePerTask
	//	 iii)	d_outputCountPerTask
	//-----------------------------------------------
	DoLog( "** Do prefix sum on output data's size");
	int *d_psKeySizes = NULL;
	cudaMalloc((void**)&d_psKeySizes, sizeof(int)*h_actualNumThreads);
	cudaMemset(d_psKeySizes, 0, sizeof(int)*h_actualNumThreads);
	int h_allKeySize = prefexSum((int*)d_outputKeysSizePerTask, (int*)d_psKeySizes, h_actualNumThreads);

	int *d_psValSizes = NULL;
	cudaMalloc((void**)&d_psValSizes, sizeof(int)*h_actualNumThreads);
	cudaMemset(d_psValSizes, 0, sizeof(int)*h_actualNumThreads);
	int h_allValSize = prefexSum((int*)d_outputValsSizePerTask, (int*)d_psValSizes, h_actualNumThreads);

	int *d_psCounts = NULL;
	cudaMalloc((void**)&d_psCounts, sizeof(int)*h_actualNumThreads);
	cudaMemset(d_psCounts, 0, sizeof(int)*h_actualNumThreads);
	int h_allCounts = prefexSum((int*)d_outputCountPerTask, (int*)d_psCounts, h_actualNumThreads);

	DoLog("** Reduce Output: key buf size %d bytes, val buf size %d bytes, index buf size %d bytes, %d records",
		h_allKeySize, h_allValSize, h_allCounts*sizeof(int4),h_allCounts);

	//-----------------------------------------------
	//7, allocate output memory on device memory
	//-----------------------------------------------
	DoLog( "** Allocate intermediate memory on device memory");
	char*	d_outputKeys = NULL;
	cudaMalloc((void**)&d_outputKeys, h_allKeySize);

	char*	d_outputVals = NULL;
	cudaMalloc((void**)&d_outputVals, h_allValSize);

	int4*	d_outputOffsetSizes = NULL;
	cudaMalloc((void**)&d_outputOffsetSizes, sizeof(int4)*h_allCounts);
	//--------------------------------------------------
	//8, start reduce
	//--------------------------------------------------
	DoLog( "** Reduce");
	
	int2*	d_keyValOffsets = NULL;
	cudaMalloc((void**)&d_keyValOffsets, sizeof(int2)*h_actualNumThreads);
	cudaMemset(d_keyValOffsets, 0, sizeof(int2)*h_actualNumThreads);

	int*	d_curIndex = NULL;
	cudaMalloc((void**)&d_curIndex, sizeof(int)*h_actualNumThreads);
	cudaMemset(d_curIndex, 0, sizeof(int)*h_actualNumThreads);
	
	int sizeSmem = h_dimBlock.x * sizeof(int) * 5;
	Reducer<<<h_dimGrid, h_dimBlock, sizeSmem>>>(d_interKeys,
									   d_interVals,
									   d_interOffsetSizes,
									   d_interKeyListRange,
									   d_psKeySizes,
									   d_psValSizes,
									   d_psCounts,
									   d_outputKeys,
									   d_outputVals,
									   d_outputOffsetSizes,
									   d_keyValOffsets,
									   d_curIndex,
									   h_interDiffKeyCount, 
									   h_recordsPerTask,
									h_actualNumThreads);
	cudaThreadSynchronize();

	//-------------------------------------------------------
	//9, copy output data to Spec_t structure
	//-------------------------------------------------------
	g_spec->outputKeys = d_outputKeys;
	g_spec->outputVals = d_outputVals;
	g_spec->outputOffsetSizes = d_outputOffsetSizes;
	g_spec->outputRecordCount = h_allCounts;
	g_spec->outputAllKeySize = h_allKeySize;
	g_spec->outputAllValSize = h_allValSize;
	
	//----------------------------------------------
	//10, free allocated memory
	//----------------------------------------------
	cudaFree(d_interKeys);
	cudaFree(d_interVals);
	cudaFree(d_interOffsetSizes);

	cudaFree(d_outputKeysSizePerTask);
	cudaFree(d_outputValsSizePerTask);
	cudaFree(d_outputCountPerTask);

	cudaFree(d_psKeySizes);
	cudaFree(d_psValSizes);
	cudaFree(d_psCounts);

	cudaFree(d_keyValOffsets);
	cudaFree(d_curIndex);	
}

//----------------------------------------------
//start main map reduce procedure
//1, init device
//2, start map
//3, start reduce
//
//param : spec
//----------------------------------------------
void MapReduce(Spec_t *spec)
{
	assert(NULL != spec);
	Spec_t* g_spec = spec;

	DoLog( "=====start map/reduce=====");

	//-------------------------------------------
	//1, init device
	//-------------------------------------------
	//CUT_DEVICE_INIT();
	DoLog( "** init GPU");
	InitMapReduce(spec);

	//-------------------------------------------
	//2, start map
	//-------------------------------------------
	DoLog( "----------start map-----------");
	if (startMap(spec))
	{
		printf("** No output.");
		return;
	}

	if (g_spec->workflow == MAP_ONLY)
	{
		g_spec->outputKeys = g_spec->interKeys;	
		g_spec->outputVals = g_spec->interVals;
		g_spec->outputOffsetSizes = g_spec->interOffsetSizes;
		g_spec->outputRecordCount = g_spec->interRecordCount;
		g_spec->outputAllKeySize = g_spec->interAllKeySize;
		g_spec->outputAllValSize = g_spec->interAllValSize;
		goto EXIT_MR;
	}

	//-------------------------------------------
	//3, start group
	//-------------------------------------------
	DoLog( "----------start group-----------");
	TimeVal_t groupTimer;
	startTimer(&groupTimer);
	startGroup(spec);
	endTimer("Group", &groupTimer);

	if (g_spec->workflow == MAP_GROUP)
	{
		g_spec->outputKeys = g_spec->interKeys;	
		g_spec->outputVals = g_spec->interVals;
		g_spec->outputOffsetSizes = g_spec->interOffsetSizes;
		g_spec->outputRecordCount = g_spec->interRecordCount;
		g_spec->outputAllKeySize = g_spec->interAllKeySize;
		g_spec->outputAllValSize = g_spec->interAllValSize;
		g_spec->outputDiffKeyCount = g_spec->interDiffKeyCount;
		if (g_spec->outputToHost == 1)
		{
			g_spec->outputKeyListRange = (int2*)malloc(sizeof(int2)*g_spec->outputDiffKeyCount);
			CUDA_SAFE_CALL(cudaMemcpy(g_spec->outputKeyListRange, g_spec->interKeyListRange, sizeof(int2)*g_spec->outputDiffKeyCount, cudaMemcpyDeviceToHost));
                        CUDA_SAFE_CALL(cudaFree(g_spec->interKeyListRange));
		}
		goto EXIT_MR;
	}

	//-------------------------------------------
	//4, start reduce
	//-------------------------------------------
	DoLog( "----------start reduce--------");
	TimeVal_t reduceTimer;
	startTimer(&reduceTimer);
	startReduce(spec);
	endTimer("Reduce", &reduceTimer);
EXIT_MR:
	if (g_spec->outputToHost == 1)
	{
		int indexSize = g_spec->outputRecordCount * sizeof(int4);
		char* h_outputKeys = (char*)malloc(g_spec->outputAllKeySize);
		if (h_outputKeys == NULL) exit(0);
		char* h_outputVals = (char*)malloc(g_spec->outputAllValSize);
		if (h_outputVals == NULL) exit(0);
		int4* h_outputOffsetSizes = (int4*)malloc(indexSize);
		if (h_outputOffsetSizes == NULL) exit(0);

		CUDA_SAFE_CALL(cudaMemcpy(h_outputKeys, g_spec->outputKeys, g_spec->outputAllKeySize, cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(h_outputVals, g_spec->outputVals, g_spec->outputAllValSize, cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(h_outputOffsetSizes, g_spec->outputOffsetSizes, indexSize, cudaMemcpyDeviceToHost));

		CUDA_SAFE_CALL(cudaFree(g_spec->outputKeys));
		CUDA_SAFE_CALL(cudaFree(g_spec->outputVals));
		CUDA_SAFE_CALL(cudaFree(g_spec->outputOffsetSizes));

		g_spec->outputKeys = h_outputKeys;
		g_spec->outputVals = h_outputVals;
		g_spec->outputOffsetSizes = h_outputOffsetSizes;
	}
}

//------------------------------------------
//the last step
//
//1, free global variables' memory
//2, close log file's file pointer
//------------------------------------------
void FinishMapReduce(Spec_t* spec)
{
	Spec_t* g_spec = spec;
	//-------------------------------------------
	//1, free global variables' memory
	//-------------------------------------------
	free(g_spec->inputKeys);
	free(g_spec->inputVals);
	free(g_spec->inputOffsetSizes);
	
	if (g_spec->outputToHost == 1)
	{
		free(g_spec->outputKeys);
		free(g_spec->outputVals);
		free(g_spec->outputOffsetSizes);
		if (g_spec->workflow == MAP_GROUP)
			free(g_spec->outputKeyListRange);
	}
	else
	{
		cudaFree(g_spec->outputKeys);
		cudaFree(g_spec->outputVals);
		cudaFree(g_spec->outputOffsetSizes);
		if (g_spec->workflow == MAP_GROUP)
			cudaFree(g_spec->outputKeyListRange);
	}

	free(g_spec);

	DoLog( "=====finish map/reduce=====");
}

#endif //__MRLIB_CU__
