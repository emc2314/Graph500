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
 *Inverted Index (II): It scans a set of HTML files
 *and extracts the positions for all links. Each Map
 *processes one line of HTML files. For each link it
 *finds, it outputs an intermediate pair with the link as the
 *key and the position as the value. No Reduce stage is
 *required.
 *****************************************************************/

#include "MarsInc.h"
#include <dirent.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "global.h"

//#define __OUTPUT__

typedef struct flist{
   char *data;
   char *name;
   int fd; 
   int size;
} filelist_t;

int count = 0;

filelist_t *makeup(char *dirname)
{
	filelist_t *filelist = NULL;
	struct dirent **namelist;

	int n = scandir(dirname, &namelist, 0, alphasort);

  	if (n < 0)
	{
	        printf("sacn dir failed!\n");
		exit(-1);
	}
	
	filelist = (filelist_t*)malloc(sizeof(filelist_t)*n);

	int i;

	for (i = 0; i < n; i++)
	{
		if (strcmp(namelist[i]->d_name, ".") != 0 &&
			strcmp(namelist[i]->d_name, "..") != 0)
		{
			filelist[count].name = strdup(dirname);
			strcat(filelist[count].name, namelist[i]->d_name);
				
			struct stat finfo;
			filelist[count].fd = open(filelist[count].name, O_RDONLY);
        		fstat(filelist[count].fd, &finfo);
    			filelist[count].size = finfo.st_size + 1;
			filelist[count].data = (char*)malloc(filelist[count].size);
			read(filelist[count].fd, filelist[count].data, finfo.st_size);
			filelist[count].data[filelist[count].size - 1] = '\0';
        		//filelist[count].data = (char*)mmap(0, finfo.st_size + 1, PROT_READ | PROT_WRITE, MAP_PRIVATE, filelist[count].fd, 0);

			//printf("%s\n", filelist[count].data);
			count++;
		}
	}
			
	return filelist;
}

void cleanup(filelist_t *filelist)
{
	int i;
	for (i = 0; i < count; i++)
	{
		free(filelist[i].name);
		munmap(filelist[i].data, filelist[i].size+1);
		//free(filelist[i].data);
        close(filelist[i].fd);
	}
	free(filelist);
}

void validate(Spec_t* spec, int num, filelist_t* filelist, char** d_data)
{
	int4* offsetSizes = (int4*)spec->outputOffsetSizes;
	int2* groupInfo = (int2*)spec->outputKeyListRange;

	for (int i = 0; i < count; i++)
		CUDA_SAFE_CALL(cudaMemcpy(filelist[i].data, d_data[i], filelist[i].size, cudaMemcpyDeviceToHost));

	if (num > spec->outputDiffKeyCount)
		num = spec->outputDiffKeyCount;

	printf("# of Groups: %d, # of records:%d\n", spec->outputDiffKeyCount, spec->outputRecordCount);
	for (int i = 0; i < num; i++)
	{
		II_KEY_T* urls = (II_KEY_T*)(spec->outputKeys + offsetSizes[groupInfo[i].x].x);
		int* fids = (int*)(spec->outputVals + offsetSizes[groupInfo[i].x].z);

		printf("========Start:%d, End:%d, URL: %s===========\n", groupInfo[i].x, groupInfo[i].y, filelist[*fids].data + urls->url_offset);
		printf("FILE LIST: ");
		int groupSize = groupInfo[i].y - groupInfo[i].x;
		for (int j = 0; j < groupSize; j++)
			printf("%s ", filelist[fids[j]].name);
		printf("\n");
	}
}

//------------------------------------------------------------------
//usage: InvertedIndex <dir> 
//param: dir the directory including HTML files
//------------------------------------------------------------------
int main( int argc, char** argv) 
{
	if (argc != 2)
	{
		printf("usage: %s <dir>\n", argv[0]);
		exit(-1);
	}

	Spec_t *spec = GetDefaultSpec();
	spec->workflow = MAP_GROUP;
#ifdef __OUTPUT__
	spec->outputToHost = 1;
#endif

	TimeVal_t alltimer;
	startTimer(&alltimer);

	//-------------------------------------------------------------
	//make map input
	//-------------------------------------------------------------
	TimeVal_t readtimer;
	startTimer(&readtimer);
	filelist_t *filelist = makeup(argv[1]);
	II_KEY_T key;
	II_VAL_T val;
	char** data = (char**)malloc(sizeof(char*)*count);
	for (int i = 0; i < count; i++)
	{
		data[i] = NULL;
		CUDA_SAFE_CALL(cudaMalloc((void**)&data[i], filelist[i].size));
		key.file_buf = data[i];
		val.file_id = i;
	
		int offset = 0;
		char* p = filelist[i].data;
		char* start = p;

		while (1)
		{
			int blockSize = 1024;
			if (offset + blockSize > filelist[i].size) blockSize = filelist[i].size - offset;
			p += blockSize;
			for (; *p != '\n' && *p != '\0'; p++);
			if (*p != '\0')
			{
				*p = '\0';
				++p;
				blockSize = (int)(p - start);
				val.block_size = blockSize;
				val.block_offset = offset;
				AddMapInputRecord(spec, &key, &val, sizeof(II_KEY_T), sizeof(II_VAL_T));	
				offset += blockSize;
				start = p;
			}	
			else
			{
				blockSize = (int)(filelist[i].size - offset);
				val.block_size = blockSize;
				val.block_offset = offset;
				AddMapInputRecord(spec, &key, &val, sizeof(II_KEY_T), sizeof(II_VAL_T));	
				break;
			}
		}	
		CUDA_SAFE_CALL(cudaMemcpy(data[i], filelist[i].data, filelist[i].size, cudaMemcpyHostToDevice));
	}
	endTimer("io-test", &readtimer); 
	//-------------------------------------------------------------
	//start MapReduce procedure
	//-------------------------------------------------------------
	MapReduce(spec);

	//-------------------------------------------------------------
	//start MapReduce procedure
	//-------------------------------------------------------------
#ifdef __OUTPUT__
	validate(spec, 2, filelist, data);
#endif

	//------------------------------------------------------------
	//finish
	//------------------------------------------------------------
	FinishMapReduce(spec);
	for (int i = 0; i < count; i++)
		CUDA_SAFE_CALL(cudaFree(data[i]));
	cleanup(filelist);
	free(data);

	endTimer("all", &alltimer);
	return 0;
}
