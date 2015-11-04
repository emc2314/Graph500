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

__device__ void MAP_COUNT_FUNC//(void *key, void *val, int keySize, int valSize)
{
	EMIT_INTER_COUNT_FUNC(sizeof(float), sizeof(int2));
}


__device__ void MAP_FUNC//(void *key, void val, int keySize, int valSize)
{
	SS_KEY_T* pKey = (SS_KEY_T*)key;
	SS_VAL_T* pVal = (SS_VAL_T*)val;

	float* matrix = pKey->matrix;
	float up = 0.0f;
	float down = 0.0f;
	float result = 0.0f;

	int doc1 = pVal->doc1;
	int doc2 = pVal->doc2;
	int M_COL_COUNT = pVal->dim;

	float4 *a = (float4*)(matrix+doc1*M_COL_COUNT);
	float4 *b = (float4*)(matrix+doc2*M_COL_COUNT);
	float doc1Down = 0.0f;
	float doc2Down = 0.0f;
	
	int col4 = M_COL_COUNT >>2;
	int remainder = M_COL_COUNT &3;
	float4 aValue, bValue;
	
	for (int i = 0; i < col4; i++)
	{
		aValue=a[i]; 
		bValue=b[i];
		up += (aValue.x *bValue.x+aValue.y *bValue.y+aValue.z *bValue.z+aValue.w *bValue.w);
	//	doc1Down += powf(aValue.x,2)+powf(aValue.y,2)+powf(aValue.z,2)+powf(aValue.w,2);
		doc1Down += aValue.x*aValue.x+aValue.y*aValue.y+aValue.z*aValue.z+aValue.w*aValue.w;
		doc2Down += bValue.x*bValue.x+bValue.y*bValue.y+bValue.z*bValue.z+bValue.w*bValue.w;
	//	doc2Down += powf(bValue.x,2)+powf(bValue.y,2)+powf(bValue.z,2)+powf(bValue.w,22);
	}
	float *a1 = (float*)(a+col4);
	float *b1 = (float*)(b+col4);

	for (int i = 0; i < remainder; i++)
	{
		float a1Value = a1[i];
		float b1Value = b1[i];
		up += (a1Value * b1Value);
		doc1Down += (a1Value*a1Value);
		doc2Down += (b1Value*b1Value);
	}
	
	down = sqrtf(doc1Down)*sqrtf(doc2Down);
	result = up / down;

	float* o_result = (float*)GET_OUTPUT_BUF(0);
	*o_result = result;
	int2* o_doc = (int2*)GET_OUTPUT_BUF(sizeof(float));
	o_doc->x = doc1;
	o_doc->y = doc2;

	EMIT_INTERMEDIATE_FUNC(o_result, o_doc, sizeof(float), sizeof(int2));			 
}

#endif //__MAP_CU__
