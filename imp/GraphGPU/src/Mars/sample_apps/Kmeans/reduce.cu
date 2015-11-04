/*$Id: reduce.cu 729 2009-11-12 09:56:09Z wenbinor $*/
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

#ifndef __REDUCE_CU__
#define __REDUCE_CU__

#include "MarsInc.h"
#include "global.h"

//-------------------------------------------------------------------------
//No Reduce in this application
//-------------------------------------------------------------------------
__device__ void REDUCE_COUNT_FUNC//(void* key, void* vals, size_t keySize, size_t valCount)
{
	EMIT_COUNT_FUNC(sizeof(KM_KEY_T), sizeof(KM_VAL_T));
}

__device__ void REDUCE_FUNC//(void* key, void* vals, size_t keySize, size_t valCount)
{
	KM_KEY_T* pKey = (KM_KEY_T*)key;
	KM_VAL_T* pFirstVal = (KM_VAL_T*)vals;
	int dim = pKey->dim;
	int firstPtId = pKey->point_id;
	int cluster_id = pKey->ptrClusterId[firstPtId];
	int* clusters = (int*)pFirstVal->ptrClusters + cluster_id * dim;
	int* points = (int*)pFirstVal->ptrPoints;

	for (int i = 0; i < dim; i++)
		clusters[i] = 0;

	for (int i = 0; i < valCount; i++)
	{
		KM_KEY_T* iKey = (KM_KEY_T*)GET_KEY_FUNC(key, i);
		int* pt = points + iKey->point_id * dim;	
		for (int j = 0; j < dim; j++)
			clusters[j] += pt[j];	
	}

	for (int i = 0; i < dim; i++)
		clusters[i] /= (int)valCount;

	//EMIT_FUNC(key, vals, sizeof(KM_KEY_T), sizeof(KM_VAL_T));
}
#endif //__REDUCE_CU__
