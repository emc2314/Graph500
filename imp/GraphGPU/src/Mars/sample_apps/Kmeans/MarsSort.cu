/*$Id: MarsSort.cu 724 2009-11-10 10:27:18Z wenbinor $*/
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

#ifndef _SORT_CU_
#define _SORT_CU_

#include "MarsInc.h"
#include "compare.cu"

#define NUM_BLOCK_PER_CHUNK_BITONIC_SORT 512//b256
#define SHARED_MEM_INT2 256
#define NUM_BLOCKS_CHUNK 256//(512)
#define	NUM_THREADS_CHUNK 256//(256)
#define CHUNK_SIZE (NUM_BLOCKS_CHUNK*NUM_THREADS_CHUNK)
#define NUM_CHUNKS_R (NUM_RECORDS_R/CHUNK_SIZE)

__device__ int getCompareValue(void *d_rawData, cmp_type_t value1, cmp_type_t value2)
{
	int compareValue=0;
	int v1=value1.x;
	int v2=value2.x;
	if((v1==-1) || (v2==-1))
	{
		if(v1==v2)
			compareValue=0;
		else
			if(v1==-1)
				compareValue=-1;
			else
				compareValue=1;
	}
	else
		compareValue=compare((void*)(((char*)d_rawData)+v1), value1.y, (void*)(((char*)d_rawData)+v2), value2.y); 

	return compareValue;
}

void * s_qsRawData=NULL;


__global__ void
partBitonicSortKernel( void* d_rawData, int totalLenInBytes,cmp_type_t* d_R, unsigned int numRecords, int chunkIdx, int unitSize)
{
	__shared__ cmp_type_t shared[NUM_THREADS_CHUNK];

	int tx = threadIdx.x;
	int bx = blockIdx.x;

	//load the data
	int dataIdx = chunkIdx*CHUNK_SIZE+bx*blockDim.x+tx;
	int unitIdx = ((NUM_BLOCKS_CHUNK*chunkIdx + bx)/unitSize)&1;
	shared[tx] = d_R[dataIdx];
	__syncthreads();
	int ixj=0;
	int a=0;
	cmp_type_t temp1;
	cmp_type_t temp2;
	int k = NUM_THREADS_CHUNK;

	if(unitIdx == 0)
	{
		for (int j = (k>>1); j>0; j =(j>>1))
		{
			ixj = tx ^ j;
			//a = (shared[tx].y - shared[ixj].y);				
			temp1=shared[tx];
			temp2= shared[ixj];
			if (ixj > tx) {
				//a=temp1.y-temp2.y;
				//a=compareString((void*)(((char4*)d_rawData)+temp1.x),(void*)(((char4*)d_rawData)+temp2.x)); 
				a=getCompareValue(d_rawData, temp1, temp2);
				if ((tx & k) == 0) {
					if ( (a>0)) {
						shared[tx]=temp2;
						shared[ixj]=temp1;
					}
				}
				else {
					if ( (a<0)) {
						shared[tx]=temp2;
						shared[ixj]=temp1;
					}
				}
			}
				
			__syncthreads();
		}
	}
	else
	{
		for (int j = (k>>1); j>0; j =(j>>1))
		{
			ixj = tx ^ j;
			temp1=shared[tx];
			temp2= shared[ixj];
			
			if (ixj > tx) {					
				//a=temp1.y-temp2.y;					
				//a=compareString((void*)(((char4*)d_rawData)+temp1.x),(void*)(((char4*)d_rawData)+temp2.x));
				a=getCompareValue(d_rawData, temp1, temp2);
				if ((tx & k) == 0) {
					if( (a<0))
					{
						shared[tx]=temp2;
						shared[ixj]=temp1;
					}
				}
				else {
					if( (a>0))
					{
						shared[tx]=temp2;
						shared[ixj]=temp1;
					}
				}
			}
			
			__syncthreads();
		}
	}

	d_R[dataIdx] = shared[tx];
}

__global__ void
unitBitonicSortKernel(void* d_rawData, int totalLenInBytes, cmp_type_t* d_R, unsigned int numRecords, int chunkIdx )
{
	__shared__ cmp_type_t shared[NUM_THREADS_CHUNK];

	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int unitIdx = (NUM_BLOCKS_CHUNK*chunkIdx + bx)&1;

	//load the data
	int dataIdx = chunkIdx*CHUNK_SIZE+bx*blockDim.x+tx;
	shared[tx] = d_R[dataIdx];
	__syncthreads();

	cmp_type_t temp1;
	cmp_type_t temp2;
	int ixj=0;
	int a=0;
	if(unitIdx == 0)
	{
		for (int k = 2; k <= NUM_THREADS_CHUNK; (k =k<<1))
		{
			// bitonic merge:
			for (int j = (k>>1); j>0; (j=j>>1))
			{
				ixj = tx ^ j;	
				temp1=shared[tx];
				temp2= shared[ixj];
				if (ixj > tx) {					
					//a=temp1.y-temp2.y;
					//a=compareString((void*)(((char4*)d_rawData)+temp1.x),(void*)(((char4*)d_rawData)+temp2.x));
					a=getCompareValue(d_rawData, temp1, temp2);
					if ((tx & k) == 0) {
						if ( (a>0)) {
							shared[tx]=temp2;
							shared[ixj]=temp1;
						}
					}
					else {
						if ( (a<0)) {
							shared[tx]=temp2;
							shared[ixj]=temp1;
						}
					}
				}
				
				__syncthreads();
			}
		}
	}
	else
	{
		for (int k = 2; k <= NUM_THREADS_CHUNK; (k =k<<1))
		{
			// bitonic merge:
			for (int j = (k>>1); j>0; (j=j>>1))
			{
				ixj = tx ^ j;
				temp1=shared[tx];
				temp2= shared[ixj];
				if (ixj > tx) {					
					//a=temp1.y-temp2.y;
					//a=compareString((void*)(((char4*)d_rawData)+temp1.x),(void*)(((char4*)d_rawData)+temp2.x));
					a=getCompareValue(d_rawData, temp1, temp2);
					if ((tx & k) == 0) {
						if( (a<0))
						{
							shared[tx]=temp2;
							shared[ixj]=temp1;
						}
					}
					else {
						if( (a>0))
						{
							shared[tx]=temp2;
							shared[ixj]=temp1;
						}
					}
				}
				
				__syncthreads();
			}
		}

	}

	d_R[dataIdx] = shared[tx];
}

__global__ void
bitonicKernel( void* d_rawData, int totalLenInBytes, cmp_type_t* d_R, unsigned int numRecords, int k, int j)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tid = threadIdx.x;
	int dataIdx = by*gridDim.x*blockDim.x + bx*blockDim.x + tid;

	int ixj = dataIdx^j;

	if( ixj > dataIdx )
	{
		cmp_type_t tmpR = d_R[dataIdx];
		cmp_type_t tmpIxj = d_R[ixj];
		if( (dataIdx&k) == 0 )
		{
			//if( tmpR.y > tmpIxj.y )
			//if(compareString((void*)(((char4*)d_rawData)+tmpR.x),(void*)(((char4*)d_rawData)+tmpIxj.x))==1) 
			if(getCompareValue(d_rawData, tmpR, tmpIxj)==1)
			{
				d_R[dataIdx] = tmpIxj;
				d_R[ixj] = tmpR;
			}
		}
		else
		{
			//if( tmpR.y < tmpIxj.y )
			//if(compareString((void*)(((char4*)d_rawData)+tmpR.x),(void*)(((char4*)d_rawData)+tmpIxj.x))==-1) 
			if(getCompareValue(d_rawData, tmpR, tmpIxj)==-1)
			{
				d_R[dataIdx] = tmpIxj;
				d_R[ixj] = tmpR;
			}
		}
	}
}

__device__ inline void swap(cmp_type_t & a, cmp_type_t & b)
{
	// Alternative swap doesn't use a temporary register:
	// a ^= b;
	// b ^= a;
	// a ^= b;
	
    cmp_type_t tmp = a;
    a = b;
    b = tmp;
}

__global__ void bitonicSortSingleBlock_kernel(void* d_rawData, int totalLenInBytes, cmp_type_t * d_values, int rLen, cmp_type_t* d_output)
{
	__shared__ cmp_type_t bs_cmpbuf[SHARED_MEM_INT2];
	

    //const int by = blockIdx.y;
	//const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	//const int bid=bx+by*gridDim.x;
	//const int numThread=blockDim.x;
	//const int resultID=(bx)*numThread+tid;
	
	if(tid<rLen)
	{
		bs_cmpbuf[tid] = d_values[tid];
	}
	else
	{
		bs_cmpbuf[tid].x =-1;
	}

    __syncthreads();

    // Parallel bitonic sort.
	int compareValue=0;
    for (int k = 2; k <= SHARED_MEM_INT2; k *= 2)
    {
        // Bitonic merge:
        for (int j = k / 2; j>0; j /= 2)
        {
            int ixj = tid ^ j;
            
            if (ixj > tid)
            {
                if ((tid & k) == 0)
                {
					compareValue=getCompareValue(d_rawData, bs_cmpbuf[tid], bs_cmpbuf[ixj]);
					//if (shared[tid] > shared[ixj])
					if(compareValue>0)
                    {
                        swap(bs_cmpbuf[tid], bs_cmpbuf[ixj]);
                    }
                }
                else
                {
					compareValue=getCompareValue(d_rawData, bs_cmpbuf[tid], bs_cmpbuf[ixj]);
                    //if (shared[tid] < shared[ixj])
					if(compareValue<0)
                    {
                        swap(bs_cmpbuf[tid], bs_cmpbuf[ixj]);
                    }
                }
            }
            
            __syncthreads();
        }
    }

    // Write result.
	/*if(tid<rLen)
	{
		d_output[tid] = bs_cmpbuf[tid+SHARED_MEM_INT2-rLen];
	}*/
	int startCopy=SHARED_MEM_INT2-rLen;
	if(tid>=startCopy)
	{
		d_output[tid-startCopy]=bs_cmpbuf[tid];
	}
}

__global__ void bitonicSortMultipleBlocks_kernel(void* d_rawData, int totalLenInBytes, cmp_type_t * d_values, int* d_bound, int startBlock, int numBlock, cmp_type_t *d_output)
{
	__shared__ int bs_pStart;
	__shared__ int bs_pEnd;
	__shared__ int bs_numElement;
    __shared__ cmp_type_t bs_shared[SHARED_MEM_INT2];
	

    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	//const int numThread=blockDim.x;
	//const int resultID=(bx)*numThread+tid;
	if(bid>=numBlock) return;

	if(tid==0)
	{
		bs_pStart=d_bound[(bid+startBlock)<<1];
		bs_pEnd=d_bound[((bid+startBlock)<<1)+1];
		bs_numElement=bs_pEnd-bs_pStart;
		//if(bid==82&& bs_pStart==6339)
		//	printf("%d, %d, %d\n", bs_pStart, bs_pEnd, bs_numElement);
		
	}
	__syncthreads();
    // Copy input to shared mem.
	if(tid<bs_numElement)
	{
		bs_shared[tid] = d_values[tid+bs_pStart];
		//if(bid==82 && bs_pStart==6339)
		//	printf("tid %d, pos, %d, %d, %d, %d\n", tid,tid+bs_pStart, bs_pStart,bs_pEnd, d_values[tid+bs_pStart].x);
		//if(6342==tid+bs_pStart)
		//	printf(")))tid %d, pos, %d, %d, %d, %d\n", tid,tid+bs_pStart, bs_pStart,bs_pEnd, d_values[tid+bs_pStart].x);
	}
	else
	{
		bs_shared[tid].x =-1;
	}

    __syncthreads();

    // Parallel bitonic sort.
	int compareValue=0;
    for (int k = 2; k <= SHARED_MEM_INT2; k *= 2)
    {
        // Bitonic merge:
        for (int j = k / 2; j>0; j /= 2)
        {
            int ixj = tid ^ j;
            
            if (ixj > tid)
            {
                if ((tid & k) == 0)
                {
					compareValue=getCompareValue(d_rawData, bs_shared[tid], bs_shared[ixj]);
					//if (shared[tid] > shared[ixj])
					if(compareValue>0)
                    {
                        swap(bs_shared[tid], bs_shared[ixj]);
                    }
                }
                else
                {
					compareValue=getCompareValue(d_rawData, bs_shared[tid], bs_shared[ixj]);
                    //if (shared[tid] < shared[ixj])
					if(compareValue<0)
                    {
                        swap(bs_shared[tid], bs_shared[ixj]);
                    }
                }
            }
            
            __syncthreads();
        }
    }

    // Write result.
	//if(tid<bs_numElement)
	//{
	//	d_output[tid+bs_pStart] = bs_shared[tid+SHARED_MEM_INT2-bs_numElement];
	//}
	//int startCopy=SHARED_MEM_INT2-bs_numElement;
	if(tid>=bs_numElement)
	{
		d_output[tid-bs_numElement]=bs_shared[tid];
	}
}


__global__ void initialize_kernel(cmp_type_t* d_data, int startPos, int rLen, cmp_type_t value)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	d_data[pos]=value;
}
void bitonicSortMultipleBlocks(void* d_rawData, int totalLenInBytes, cmp_type_t * d_values, int* d_bound, int numBlock, cmp_type_t * d_output)
{
	int numThreadsPerBlock_x=SHARED_MEM_INT2;
	int numThreadsPerBlock_y=1;
	int numBlock_x=NUM_BLOCK_PER_CHUNK_BITONIC_SORT;
	int numBlock_y=1;
	int numChunk=numBlock/numBlock_x;
	if(numBlock%numBlock_x!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*numBlock_x;
		end=start+numBlock_x;
		if(end>numBlock)
			end=numBlock;
		//printf("bitonicSortMultipleBlocks_kernel: %d, range, %d, %d\n", i, start, end);
		bitonicSortMultipleBlocks_kernel<<<grid,thread>>>(d_rawData, totalLenInBytes, d_values, d_bound, start, end-start, d_output);
		cudaThreadSynchronize();
	}
//	cudaThreadSynchronize();
}


void bitonicSortSingleBlock(void* d_rawData, int totalLenInBytes, cmp_type_t * d_values, int rLen, cmp_type_t * d_output)
{
	int numThreadsPerBlock_x=SHARED_MEM_INT2;
	int numThreadsPerBlock_y=1;
	int numBlock_x=1;
	int numBlock_y=1;
	

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	bitonicSortSingleBlock_kernel<<<grid,thread>>>(d_rawData, totalLenInBytes, d_values, rLen, d_output);
	cudaThreadSynchronize();
}



void initialize(cmp_type_t *d_data, int rLen, cmp_type_t value)
{
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		initialize_kernel<<<grid,thread>>>(d_data, start, rLen, value);
	} 
	cudaThreadSynchronize();
}
void bitonicSortGPU(void* d_rawData, int totalLenInBytes, cmp_type_t* d_Rin, int rLen, void *d_Rout)
{
	unsigned int numRecordsR;

	unsigned int size = rLen;
	unsigned int level = 0;
	while( size != 1 )
	{
		size = size/2;
		level++;
	}

	if( (1<<level) < rLen )
	{
		level++;
	}

	numRecordsR = (1<<level);
	if(rLen<=NUM_THREADS_CHUNK)
	{
		bitonicSortSingleBlock((void*)d_rawData, totalLenInBytes, d_Rin, rLen, (cmp_type_t*)d_Rout);
	}
	else
	if( rLen <= 256*1024 )
	{
		//unsigned int numRecordsR = rLen;
		
		unsigned int numThreadsSort = NUM_THREADS_CHUNK;
		if(numRecordsR<NUM_THREADS_CHUNK)
			numRecordsR=NUM_THREADS_CHUNK;
		unsigned int numBlocksXSort = numRecordsR/numThreadsSort;
		unsigned int numBlocksYSort = 1;
		dim3 gridSort( numBlocksXSort, numBlocksYSort );		
		unsigned int memSizeRecordsR = sizeof( cmp_type_t ) * numRecordsR;
		//copy the <offset, length> pairs.
		cmp_type_t* d_R;
		CUDA_SAFE_CALL( cudaMalloc( (void**) &d_R, memSizeRecordsR) );
		cmp_type_t tempValue;
		tempValue.x=tempValue.y=-1;
		initialize(d_R, numRecordsR, tempValue);
		CUDA_SAFE_CALL( cudaMemcpy( d_R, d_Rin, rLen*sizeof(cmp_type_t), cudaMemcpyDeviceToDevice) );
		

		for( int k = 2; k <= numRecordsR; k *= 2 )
		{
			for( int j = k/2; j > 0; j /= 2 )
			{
				bitonicKernel<<<gridSort, numThreadsSort>>>((void*)d_rawData, totalLenInBytes, d_R, numRecordsR, k, j);
			}
		}
		CUDA_SAFE_CALL( cudaMemcpy( d_Rout, d_R+(numRecordsR-rLen), sizeof(cmp_type_t)*rLen, cudaMemcpyDeviceToDevice) );
		cudaFree( d_R );
		cudaThreadSynchronize();
	}
	else
	{
		unsigned int numThreadsSort = NUM_THREADS_CHUNK;
		unsigned int numBlocksYSort = 1;
		unsigned int numBlocksXSort = (numRecordsR/numThreadsSort)/numBlocksYSort;
		if(numBlocksXSort>=(1<<16))
		{
			numBlocksXSort=(1<<15);
			numBlocksYSort=(numRecordsR/numThreadsSort)/numBlocksXSort;			
		}
		unsigned int numBlocksChunk = NUM_BLOCKS_CHUNK;
		unsigned int numThreadsChunk = NUM_THREADS_CHUNK;
		
		unsigned int chunkSize = numBlocksChunk*numThreadsChunk;
		unsigned int numChunksR = numRecordsR/chunkSize;

		dim3 gridSort( numBlocksXSort, numBlocksYSort );
		unsigned int memSizeRecordsR = sizeof( cmp_type_t ) * numRecordsR;

		cmp_type_t* d_R;
		CUDA_SAFE_CALL( cudaMalloc( (void**) &d_R, memSizeRecordsR) );
		cmp_type_t tempValue;
		tempValue.x=tempValue.y=-1;
		initialize(d_R, numRecordsR, tempValue);
		CUDA_SAFE_CALL( cudaMemcpy( d_R, d_Rin, rLen*sizeof(cmp_type_t), cudaMemcpyDeviceToDevice) );

		for( int chunkIdx = 0; chunkIdx < numChunksR; chunkIdx++ )
		{
			unitBitonicSortKernel<<< numBlocksChunk, numThreadsChunk>>>( (void*)d_rawData, totalLenInBytes, d_R, numRecordsR, chunkIdx );
		}

		int j;
		for( int k = numThreadsChunk*2; k <= numRecordsR; k *= 2 )
		{
			for( j = k/2; j > numThreadsChunk/2; j /= 2 )
			{
				bitonicKernel<<<gridSort, numThreadsSort>>>( (void*)d_rawData, totalLenInBytes, d_R, numRecordsR, k, j);
			}

			for( int chunkIdx = 0; chunkIdx < numChunksR; chunkIdx++ )
			{
				partBitonicSortKernel<<< numBlocksChunk, numThreadsChunk>>>((void*)d_rawData, totalLenInBytes, d_R, numRecordsR, chunkIdx, k/numThreadsSort );
			}
		}
		CUDA_SAFE_CALL( cudaMemcpy( d_Rout, d_R+(numRecordsR-rLen), sizeof(cmp_type_t)*rLen, cudaMemcpyDeviceToDevice) );
		cudaFree( d_R );
		cudaThreadSynchronize();
	}
}

__global__ void getIntYArray_kernel(int2* d_input, int startPos, int rLen, int* d_output)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		int2 value=d_input[pos];
		d_output[pos]=value.y;
	}
}


__global__ void getXYArray_kernel(cmp_type_t* d_input, int startPos, int rLen, int2* d_output)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		cmp_type_t value=d_input[pos];
		d_output[pos].x=value.x;
		d_output[pos].y=value.y;
	}
}

__global__ void getZWArray_kernel(cmp_type_t* d_input, int startPos, int rLen, int2* d_output)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		cmp_type_t value=d_input[pos];
		d_output[pos].x=value.z;
		d_output[pos].y=value.w;
	}
}


__global__ void setXYArray_kernel(cmp_type_t* d_input, int startPos, int rLen, int2* d_value)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		cmp_type_t value=d_input[pos];
		value.x=d_value[pos].x;
		value.y=d_value[pos].y;
		d_input[pos]=value;
	}
}

__global__ void setZWArray_kernel(cmp_type_t* d_input, int startPos, int rLen, int2* d_value)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		cmp_type_t value=d_input[pos];
		value.z=d_value[pos].x;
		value.w=d_value[pos].y;
		d_input[pos]=value;
	}
}

void getIntYArray(int2 *d_data, int rLen, int* d_output)
{
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		getIntYArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_output);
	} 
	cudaThreadSynchronize();
}

void getXYArray(cmp_type_t *d_data, int rLen, int2* d_output)
{
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		getXYArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_output);
	} 
	cudaThreadSynchronize();
}

void getZWArray(cmp_type_t *d_data, int rLen, int2* d_output)
{
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		getZWArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_output);
	} 
	cudaThreadSynchronize();
}



void setXYArray(cmp_type_t *d_data, int rLen, int2* d_value)
{
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		setXYArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_value);
	} 
	cudaThreadSynchronize();
}

void setZWArray(cmp_type_t *d_data, int rLen, int2* d_value)
{
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		setZWArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_value);
	} 
	cudaThreadSynchronize();
}
__global__ void copyChunks_kernel(void *d_source, int startPos, int2* d_Rin, int rLen, int *d_sum, void *d_dest)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	
	if(pos<rLen)
	{
		int2 value=d_Rin[pos];
		int offset=value.x;
		int size=value.y;
		int startWritePos=d_sum[pos];
		int i=0;
		char *source=(char*)d_source;
		char *dest=(char*)d_dest;
		for(i=0;i<size;i++)
		{
			dest[i+startWritePos]=source[i+offset];
		}
		value.x=startWritePos;
		d_Rin[pos]=value;
	}
}

__global__ void getChunkBoundary_kernel(void* d_rawData, int startPos, cmp_type_t *d_Rin, 
										int rLen, int* d_startArray)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	
	if(pos<rLen)
	{
		int result=0;
		if(pos==0)//the start position
		{
			result=1;
		}
		else
		{
			cmp_type_t cur=d_Rin[pos];
			cmp_type_t left=d_Rin[pos-1];
			if(getCompareValue(d_rawData, cur, left)!=0)
			{
				result=1;
			}
		}
		d_startArray[pos]=result;	
	}
}

__global__ void setBoundaryInt2_kernel(int* d_boundary, int startPos, int numKey, int rLen,
										  int2* d_boundaryRange)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	
	if(pos<numKey)
	{
		int2 flag;
		flag.x=d_boundary[pos];
		if((pos+1)!=numKey)
			flag.y=d_boundary[pos+1];
		else
			flag.y=rLen;
		d_boundaryRange[pos]=flag;
	}
}

__global__ void writeBoundary_kernel(int startPos, int rLen, int* d_startArray,
									int* d_startSumArray, int* d_bounary)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	
	if(pos<rLen)
	{
		int flag=d_startArray[pos];
		int writePos=d_startSumArray[pos];
		if(flag==1)
			d_bounary[writePos]=pos;
	}
}

void copyChunks(void *d_source, int2* d_Rin, int rLen, void *d_dest)
{
	//extract the size information for each chunk
	int* d_size;
	CUDA_SAFE_CALL( cudaMalloc( (void**) (&d_size), sizeof(int)*rLen) );	
	getIntYArray(d_Rin, rLen, d_size);
	//compute the prefix sum for the output positions.
	int* d_sum;
	CUDA_SAFE_CALL( cudaMalloc( (void**) (&d_sum), sizeof(int)*rLen) );
	saven_initialPrefixSum(rLen);
	prescanArray(d_sum,d_size,rLen);
	cudaFree(d_size);
	//output
	int numThreadsPerBlock_x=128;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		copyChunks_kernel<<<grid,thread>>>(d_source, start, d_Rin, rLen, d_sum, d_dest);
	} 
	cudaThreadSynchronize();
	
	cudaFree(d_sum);
	
}
//return the number of chunks.
int getChunkBoundary(void *d_source, cmp_type_t* d_Rin, int rLen, int2 ** h_outputKeyListRange)
{
	int resultNumChunks=0;
	//get the chunk boundary[start of chunk0, start of chunk 1, ...]
	int* d_startArray;
	CUDA_SAFE_CALL( cudaMalloc( (void**) (&d_startArray), sizeof(int)*rLen) );	
	
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		getChunkBoundary_kernel<<<grid,thread>>>(d_source, start, d_Rin, rLen, d_startArray);
	} 
	cudaThreadSynchronize();
	//prefix sum for write positions.
	int* d_startSumArray;
	CUDA_SAFE_CALL( cudaMalloc( (void**) (&d_startSumArray), sizeof(int)*rLen) );
	saven_initialPrefixSum(rLen);
	prescanArray(d_startSumArray,d_startArray,rLen);

	//gpuPrint(d_startSumArray, rLen, "d_startSumArray");

	int lastValue=0;
	int partialSum=0;
	CUDA_SAFE_CALL( cudaMemcpy( &lastValue, d_startArray+(rLen-1), sizeof(int), cudaMemcpyDeviceToHost) );
	//gpuPrint(d_startArray, rLen, "d_startArray");
	CUDA_SAFE_CALL( cudaMemcpy( &partialSum, d_startSumArray+(rLen-1), sizeof(int), cudaMemcpyDeviceToHost) );
	//gpuPrint(d_startSumArray, rLen, "d_startSumArray");
	resultNumChunks=lastValue+partialSum;

	int* d_boundary;//[start of chunk0, start of chunk 1, ...]
	CUDA_SAFE_CALL( cudaMalloc( (void**) (&d_boundary), sizeof(int)*resultNumChunks) );

	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		writeBoundary_kernel<<<grid,thread>>>(start, rLen, d_startArray,
									d_startSumArray, d_boundary);
	} 
	cudaFree(d_startArray);
	cudaFree(d_startSumArray);	

	//set the int2 boundary. 
	int2 *d_outputKeyListRange;
	CUDA_SAFE_CALL( cudaMalloc( (void**) (&d_outputKeyListRange), sizeof(int2)*resultNumChunks) );
	numChunk=resultNumChunks/chunkSize;
	if(resultNumChunks%chunkSize!=0)
		numChunk++;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>resultNumChunks)
			end=resultNumChunks;
		setBoundaryInt2_kernel<<<grid,thread>>>(d_boundary, start, resultNumChunks, rLen, d_outputKeyListRange);
	} 
	cudaThreadSynchronize();

	*h_outputKeyListRange=(int2*)malloc(sizeof(int2)*resultNumChunks);
	CUDA_SAFE_CALL( cudaMemcpy( *h_outputKeyListRange, d_outputKeyListRange, sizeof(int2)*resultNumChunks, cudaMemcpyDeviceToHost) );
	
	cudaFree(d_boundary);
	cudaFree(d_outputKeyListRange);
	return resultNumChunks;

}

int sort_GPU (void * d_inputKeyArray, int totalKeySize, void * d_inputValArray, int totalValueSize, 
		  cmp_type_t * d_inputPointerArray, int rLen, 
		  void * d_outputKeyArray, void * d_outputValArray, 
		  cmp_type_t * d_outputPointerArray, int2 ** h_outputKeyListRange
		  )
{
	//array_startTime(1);
	int numDistinctKey=0;
	int totalLenInBytes=-1;
	bitonicSortGPU(d_inputKeyArray, totalLenInBytes, d_inputPointerArray, rLen, d_outputPointerArray);
	//array_endTime("sort", 1);
	//!we first scatter the values and then the keys. so that we can reuse d_PA. 
	int2 *d_PA;
	CUDA_SAFE_CALL( cudaMalloc( (void**) (&d_PA), sizeof(int2)*rLen) );	
	//scatter the values.
	if(d_inputValArray!=NULL)
	{
		getZWArray(d_outputPointerArray, rLen, d_PA);
		copyChunks(d_inputValArray, d_PA, rLen, d_outputValArray);
		setZWArray(d_outputPointerArray, rLen, d_PA);
	}
	
	//scatter the keys.
	if(d_inputKeyArray!=NULL)
	{
		getXYArray(d_outputPointerArray, rLen, d_PA);
		copyChunks(d_inputKeyArray, d_PA, rLen, d_outputKeyArray);	
		setXYArray(d_outputPointerArray, rLen, d_PA);
	}
	//find the boudary for each key.

	numDistinctKey=getChunkBoundary(d_outputKeyArray, d_outputPointerArray, rLen, h_outputKeyListRange);

	return numDistinctKey;

}

#endif 
