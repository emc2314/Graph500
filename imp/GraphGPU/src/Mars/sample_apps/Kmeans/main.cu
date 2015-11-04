/*$Id: main.cu 729 2009-11-12 09:56:09Z wenbinor $*/
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
 * Kmeans (KM): It implements the iterative K-means algorithm [21] that groups a set of
 * input data points into clusters. The MapReduce procedure is called iteratively until the result
 * converges. In each iteration, each Map takes the pointer that points to centers of existing clusters
 * as the key and a point as the value. It calculates the distance between each point and each cluster
 * center, then assigns the point to the closest cluster. For each point, it emits the cluster id as the
 * key and the point as the value. The Reduce task gathers all points with the same cluster-id, and
 * calculates the cluster center. It emits the cluster id as the key and the cluster center as the value.
 * Each point is represented as a 3-dimensional real number vector. The number of clusters is 24.
 ******************************************************************/

#include "MarsInc.h"
#include "global.h"

//#define __OUTPUT__

static int *GenPoints(int numPt, int dim)
{
	int *matrix = (int*)malloc(sizeof(int)*numPt*dim);

	srand(time(0));
	for (int i = 0; i < numPt; i++)
		for (int j = 0; j < dim; j++)
			matrix[i*dim+j] = (int)(rand() % 100);

	return matrix;
}

static int* GenInitCenters(int* points, int numPt, int dim, int K)
{
	int* centers = (int*)malloc(sizeof(int)*K*dim);

	for (int i = 0; i < K; i++)
		for (int j = 0; j < dim; j++)
			centers[i*dim+j] = points[i*dim + j];
	return centers;
}

void printFun(void* key, void* val, int keySize, int valSize)
{
}

void validate(int* points, int* clusters, int ptNum, int dim, int K, Spec_t* spec, int num)
{
	int* clusterId = (int*)malloc(sizeof(int)*ptNum);
	memset(clusterId, 0, sizeof(int)*ptNum);

	int iter = 0;
	while (1)
	{
		int change = 0;
		printf("========iteration:%d===========\n", iter);
		for (int i = 0; i < ptNum; i++)
		{
			int minMean = 0;
			int* curPoint = points + dim * i;
			int* originCluster = clusters + clusterId[i] * dim; 
			for (int j = 0; j < dim; ++j)
				minMean += (originCluster[j] - curPoint[j])* (originCluster[j] - curPoint[j]);

			int curClusterId = clusterId[i];
			for (int k = 0; k < K; ++k)
			{
				int* curCluster = clusters + k*dim;
				int curMean = 0;
				for (int x = 0; x < dim; ++x)
					curMean += (curCluster[x] - curPoint[x]) * (curCluster[x] - curPoint[x]);

//printf("pt:%d, cl:%d, minDist:%d, curDist:%d\n", curPoint[0], curCluster[0], minMean, curMean);
				if (minMean > curMean) 
				{
					curClusterId = k;	
					minMean = curMean;
				}
			}

//printf("point:%d, curClusterId:%d, clusterId[i]:%d\n", i, curClusterId, clusterId[i]);
			if (curClusterId != clusterId[i]) 
			{
				change = 1;
				clusterId[i] = curClusterId;
			}
		}
		if (change == 0) break;

		int* tmpClusters = (int*)malloc(sizeof(int)*K*dim);
		memset(tmpClusters, 0, sizeof(int)*K*dim);

		int* counter = (int*)malloc(sizeof(int)*K);
		memset(counter, 0, K*sizeof(int));	
		for (int i = 0; i < ptNum; i++)
		{
			for (int j = 0; j < dim; j++)
				tmpClusters[clusterId[i] * dim + j] += points[i*dim + j];
			counter[clusterId[i]]++;
		}
		for (int i = 0; i < K; i++)
		{
			printf("cluster %d: ", i);
			if (counter[i] !=0)
			{
				for (int j =0; j < dim; j++)
				{
					tmpClusters[i*dim + j] /= counter[i];
					clusters[i*dim + j] = tmpClusters[i*dim + j];
				}
			}
			for (int j = 0; j < dim; j++)
				printf("%d ", clusters[i*dim + j]);
			printf("\n");
		}

		free(tmpClusters);
		free(counter);
		iter++;
	}//while

	free(clusterId);
}

//-----------------------------------------------------------------------
//usage: Kmeans numPt dim K
//param: numPt -- the number of points
//param: dim -- the dimension of points
//param: K -- the number of clusters
//-----------------------------------------------------------------------
int main( int argc, char** argv) 
{
	if (argc != 4)
	{
		printf("usage: %s numPt dim K\n", argv[0]);
		exit(-1);	
	}
	
	Spec_t *spec = GetDefaultSpec();
	spec->workflow = MAP_REDUCE;

	int numPt = atoi(argv[1]);
	int dim = atoi(argv[2]);
	int K = atoi(argv[3]);

	int* h_points = GenPoints(numPt, dim);
	int* h_cluster = GenInitCenters(h_points, numPt, dim, K);

	int* d_points = NULL;
	int* d_cluster = NULL;
	int* d_change = NULL;
	int* d_clusterId = NULL;
	
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_points, numPt*dim*sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_points, h_points, numPt*dim*sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_clusterId, numPt*sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(d_clusterId, 0, numPt*sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_cluster, K*dim*sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_cluster, h_cluster, K*dim*sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_change, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(d_change, 0, sizeof(int)));

	KM_VAL_T val;
	val.ptrPoints = d_points;
	val.ptrClusters = d_cluster;
	val.ptrChange = d_change;

	KM_KEY_T key;
	key.dim = dim;
	key.K = K;
	key.ptrClusterId = d_clusterId;

	TimeVal_t allTimer;
	startTimer(&allTimer);

	TimeVal_t preTimer;
	startTimer(&preTimer);
	for (int i = 0; i < numPt; i++)
	{
		key.point_id = i;
		AddMapInputRecord(spec, &key, &val, sizeof(KM_KEY_T), sizeof(KM_VAL_T));	
	}
	endTimer("preprocess", &preTimer);

	//----------------------------------------------
	//map/reduce
	//----------------------------------------------
#ifdef __OUTPUT__
	printf("CPU-----\n");
	validate(h_points, h_cluster, numPt, dim, K, spec, 10);
#endif

#ifdef __OUTPUT__
	int iter = 0;
	printf("GPU-----\n");
#endif
	while (1)
	{
		int h_change = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_change, &h_change, sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(&h_change, d_change, sizeof(int), cudaMemcpyDeviceToHost));

#ifdef __OUTPUT__
		printf("==========iter:%d==========\n", iter);
#endif
		MapReduce(spec);
		CUDA_SAFE_CALL(cudaMemcpy(&h_change, d_change, sizeof(int), cudaMemcpyDeviceToHost));
		if (h_change == 0) break;
		
		//cudaFree(spec->outputKeys);
		//cudaFree(spec->outputVals);
		//cudaFree(spec->outputOffsetSizes);
#ifdef __OUTPUT__
		printf("==========iter:%d==========\n", iter);
		int* c = (int*)malloc(dim*K*sizeof(int));
		cudaMemcpy(c, d_cluster, sizeof(int)*K*dim, cudaMemcpyDeviceToHost);
		for (int i = 0; i < K; i++)
		{
			printf("cluster:%d: ", i);
			for (int j = 0; j < dim; j++)
				printf("%d ", c[i*dim + j]);
			printf("---\n");
		}
		free(c);
		iter++;
#endif
	}

	//----------------------------------------------
	//further processing
	//----------------------------------------------

	//----------------------------------------------
	//finish
	//----------------------------------------------
	FinishMapReduce(spec);
	free(h_points);
	free(h_cluster);
	CUDA_SAFE_CALL(cudaFree(d_points));
	CUDA_SAFE_CALL(cudaFree(d_cluster));
	
	endTimer("all", &allTimer);
	return 0;
}
