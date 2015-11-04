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

/*********************************************************************************
 *Similarity Score (SS): It is used in web document clustering. 
 *The characteristics of a document are represented as a feature vector. 
 *Given two document features, a and b , the similarity score between these two
 *documents is defined to be a.b/(|a|.|b|).
 *This application computes the pair-wise similarity score for a set of documents. 
 *Each Map computes the similarity score for two documents. 
 *It outputs the intermediate pair of the score as the key and the pair of the two 
 *document IDs as the value. No Reduce stage is required.
 *********************************************************************************/

#include "MarsInc.h"
#include "global.h"

//#define __OUTPUT__

static float *GenMatrix(int M_ROW_COUNT, int M_COL_COUNT)
{
	float *matrix = (float*)malloc(sizeof(float)*M_ROW_COUNT*M_COL_COUNT);

	srand(time(0));
	for (int i = 0; i < M_ROW_COUNT; i++)
		for (int j = 0; j < M_COL_COUNT; j++)
			matrix[i*M_COL_COUNT+j] = (float)(rand() % 100);

	return matrix;
}


#define SMALL_NUM	0.0000001
int cmp(const void* a, const void* b)
{
	float aa = ((SS_VAL_T*)a)->result;
	float bb = ((SS_VAL_T*)b)->result;
	if (abs(aa-bb) < SMALL_NUM) return 0;
	if (aa < bb) return 1;
	return -1;
}


void printFun(void* key, void* val, int keySize, int valSize)
{
	float* result = (float*)key;
	int2* pos = (int2*)val;

	printf("GPU:%f - (%d, %d)\n", *result, pos->x, pos->y);
}

void validate(float* matrix, Spec_t* spec, int row_num, int col_num)
{
	SS_VAL_T* result = (SS_VAL_T*)malloc(row_num*row_num*sizeof(SS_VAL_T));
	int count = 0;
	for (int i = 0; i < row_num; i++)
	{
		int doc1 = i;
		for (int j = i+1; j < row_num; j++)
		{
             	     float up = 0;
      	             float downa = 0;
   		     float downb = 0;

			int doc2 = j;
        		for (int k= 0; k < col_num; k++){
         			up +=       matrix[doc1 *col_num + k] * matrix[doc2 *col_num + k];
         			downa += matrix[doc1 *col_num + k] * matrix[doc1 *col_num + k];
         			downb += matrix[doc2 *col_num + k] * matrix[doc2 *col_num + k];
        		}
			result[count].doc1 = doc1;
			result[count].doc2 = doc2;
       			result[count++].result = up/(sqrtf(downa)*sqrtf(downb));
		}
	}

	qsort(result, count, sizeof(SS_VAL_T), cmp);

	int displayNum  = 10;
	
	for (int i =0; i < displayNum; i++)
		printf("CPU:%f- (%d, %d)\n", result[i].result, result[i].doc1, result[i].doc2);

	printf("-------------------\n");
	PrintOutputRecords(spec, displayNum, printFun);
	free(result);
}

//--------------------------------------------------------------------
//usage: SimilarityScore rowNum colNum 
//param: rowNum
//param: colNum
//--------------------------------------------------------------------
int main( int argc, char** argv) 
{
	if (argc != 3)
	{
		printf("usage: %s rowNum colNum\n", argv[0]);
		exit(0);
	}

	Spec_t *spec = GetDefaultSpec();
#ifdef __OUTPUT__
	spec->outputToHost = 1;
#endif
	spec->workflow = MAP_GROUP;

	int M_ROW_COUNT = atoi(argv[1]);
	int M_COL_COUNT = atoi(argv[2]);
	
	//----------------------------------------------------------
	//load matrix
	//----------------------------------------------------------
	float *matrix = GenMatrix(M_ROW_COUNT, M_COL_COUNT);

	DoLog("load matrice...");

	int matrixSize = sizeof(float)*M_ROW_COUNT*M_COL_COUNT;

	float *d_matrix = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_matrix, matrixSize));
	CUDA_SAFE_CALL(cudaMemcpy(d_matrix, matrix, matrixSize, cudaMemcpyHostToDevice));

	//-----------------------------------------------------------
	//make map input
	//-----------------------------------------------------------
	SS_KEY_T ptr;
	ptr.matrix = d_matrix;
	SS_VAL_T doc_info;
	doc_info.dim = M_COL_COUNT;
	doc_info.result = 0.0f;
	TimeVal_t alltimer;
	startTimer(&alltimer);
	for (int i = 0; i < M_ROW_COUNT; i++)
	{
		doc_info.doc1 = i;

		for (int j = i+1; j < M_ROW_COUNT; j++)
		{
			doc_info.doc2 = j;
			AddMapInputRecord(spec, &ptr, &doc_info, 
				sizeof(SS_KEY_T), sizeof(SS_VAL_T));	
		}
	}

	//------------------------------------------------------------
	//main MapReduce procedure
	//------------------------------------------------------------
	MapReduce(spec);

	endTimer("all-test", &alltimer);

	//------------------------------------------------------------
	//further process
	//**Please turn on spec->outputToHost
	//------------------------------------------------------------
#ifdef __OUTPUT__
	validate(matrix, spec, M_ROW_COUNT, M_COL_COUNT);
#endif

	//------------------------------------------------------------
	//finish
	//------------------------------------------------------------
	FinishMapReduce(spec);

	return 0;
}
