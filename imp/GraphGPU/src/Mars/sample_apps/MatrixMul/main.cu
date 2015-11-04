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
 *Matrix Multiplication (MM): Matrix
 *multiplication is widely applicable to analyze the
 *relationship of two documents. Given two matrices M
 *and N, each Map computes multiplication for a row
 *from M and a column from N. It outputs the pair of the
 *row ID and the column ID as the key and the
 *corresponding result as the value. No Reduce stage is
 *required.
 ***********************************************************************/

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

static float *RotateMatrix(float *matrix, int rowCount, int colCount)
{
	float *m = (float*)malloc(sizeof(float)*rowCount*colCount);

	for (int i = 0; i < rowCount; i++)
		for (int j = 0; j < colCount; j++)
				m[i * colCount + j] = matrix[i + colCount * j];

	return m;
}

void printFun(void* key, void* val, int keySize, int valSize)
{
	float* result = (float*)key;
	int2* pos = (int2*)val;

	printf("GPU: %f - (%d, %d)\n", *result, pos->x, pos->y);
}

void validate(float* matrix1, float* matrix2, int rowCount, int colCount, Spec_t* spec, int num)
{
	float* result = (float*)malloc(rowCount*rowCount*sizeof(float));
	int2* pos = (int2*)malloc(rowCount*rowCount*sizeof(int2));
	int count = 0;
	for (int i = 0; i < rowCount; i++)
	{
		for (int k = 0; k < rowCount; k++)
		{
			result[count] = 0.0f;
			for (int j = 0; j < colCount; j++)
				result[count] += (matrix1[i*colCount+j] * matrix2[k*colCount+j]);
			pos[count].x = i;
			pos[count].y = k;
			++count;
		}
	}

	for (int i = 0; i < num; i++)
		printf("CPU:%f - (%d, %d)\n", result[i], pos[i].x, pos[i].y);
	printf("-----------------\n");
	PrintOutputRecords(spec, num, printFun);
	free(result);
	free(pos);
}

//---------------------------------------------------------------
//usage: MatrixMul rowNum colNum [dimGrid dimBlock]
//param: rowNum
//param: colNum
//---------------------------------------------------------------
int main( int argc, char** argv) 
{
	if (argc != 3)
	{	
		printf("usage: %s rowNum colNum\n", argv[0]);
		exit(-1);
	}

	Spec_t *spec = GetDefaultSpec();
	spec->workflow = MAP_ONLY;
#ifdef __OUTPUT__
	spec->outputToHost = 1;
#endif

	int M_ROW_COUNT	= atoi(argv[1]);
	int M_COL_COUNT = atoi(argv[2]);

	//------------------------------------------------------------------
	//make map input
	//------------------------------------------------------------------
	TimeVal_t timer;
	startTimer(&timer);
	printf("generate two %dx%d matrice...\n", M_ROW_COUNT, M_COL_COUNT);
	float *matrix1 =  GenMatrix(M_ROW_COUNT, M_COL_COUNT);
	float *tmpMatrix2 =  GenMatrix(M_COL_COUNT, M_ROW_COUNT);

	TimeVal_t rotateTimer;
	startTimer(&rotateTimer);
	float *matrix2 = RotateMatrix(tmpMatrix2, M_COL_COUNT, M_ROW_COUNT);
	endTimer("rotate matrix2", &rotateTimer);

	DoLog("** load matrice...");
	int matrixSize = sizeof(float)*M_ROW_COUNT*M_COL_COUNT;

	float *d_matrix1 = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_matrix1, matrixSize));
	CUDA_SAFE_CALL(cudaMemcpy(d_matrix1, matrix1, matrixSize, cudaMemcpyHostToDevice));

	float *d_matrix2 = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_matrix2, matrixSize));
	CUDA_SAFE_CALL(cudaMemcpy(d_matrix2, matrix2, matrixSize, cudaMemcpyHostToDevice));

	MM_KEY_T key;
	MM_VAL_T val;

	key.matrix1 = d_matrix1;
	key.matrix2 = d_matrix2;

	val.row_dim = M_ROW_COUNT;
	val.col_dim = M_COL_COUNT;

	startTimer(&timer);

	for (int i = 0; i < M_ROW_COUNT; i++)
	{
		val.row = i;
		for (int j = 0; j < M_ROW_COUNT; j++)
		{
			val.col = j;
			AddMapInputRecord(spec, &key, &val, sizeof(MM_KEY_T), sizeof(MM_VAL_T));	
		}
	}

	//-----------------------------------------------------------------------
	//main MapReduce procedure
	//-----------------------------------------------------------------------
	MapReduce(spec);
	endTimer("all-test", &timer);

	//-----------------------------------------------------------------------
	//further processing
	//-----------------------------------------------------------------------
#ifdef __OUTPUT__
	validate(matrix1, matrix2, M_ROW_COUNT, M_COL_COUNT,spec, 10);
#endif

	//-----------------------------------------------------------------------
	//finish
	//-----------------------------------------------------------------------
	free(matrix1);
	free(matrix2);
	free(tmpMatrix2);
	CUDA_SAFE_CALL(cudaFree(d_matrix1));
	CUDA_SAFE_CALL(cudaFree(d_matrix2));

	FinishMapReduce(spec);

	return 0;
}
