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

/********************************************************************
 *Page View Rank (PVR): With the output of the
 *Page View Count, the Map in Page View Rank takes
 *the pair of the page access count as the key and the
 *URL as the value, and obtains the top ten URLs that are
 *most frequently accessed. No Reduce stage is required.
 ********************************************************************/

#include "MarsInc.h"

#define __OUTPUT__

void printFun(void* key, void* val, int keySize, int valSize)
{
	int count = *(int*)key;
	int offset = *(int*)val;

	printf("count: %d, offset: %d\n", count, offset);
}

void validate(Spec_t* spec, int num)
{
	PrintOutputRecords(spec, num, printFun);
}

//-----------------------------------------------------------------
//usage: PageViewRank datafile 
//param: datafile
//-----------------------------------------------------------------
int main( int argc, char** argv) 
{
	if (argc != 2) 
	{
		printf("usage: %s filename\n", argv[0]);
		exit(-1);
	}

	TimeVal_t timer;
	startTimer(&timer);

	Spec_t *spec = GetDefaultSpec();
	spec->workflow = MAP_GROUP;
#ifdef __OUTPUT__
	spec->outputToHost = 1;
#endif
		
	//-----------------------------------------------------
	//make map input
	//-----------------------------------------------------
	TimeVal_t loadtimer;
	startTimer(&loadtimer);
	char *filename = argv[1];
	FILE* fp = fopen(filename, "r");
	fseek(fp, 0, SEEK_END);
	int fileSize = ftell(fp) + 1;
	rewind(fp);
    	char *h_filebuf = (char*)malloc(fileSize);
	fread(h_filebuf, fileSize, 1, fp);
	fclose(fp);

	int offset = 0;
	char* p = h_filebuf;
	char* start = h_filebuf;
	int cur = 0;
	while (1)
	{
		int lineSize = 0;
		for (; *p != '\t'; ++p, ++lineSize, ++cur);
		char* rankString = p + 1;
		for (; *p != '\n' && *p != '\0'; ++p, ++lineSize, ++cur);	

		*p = '\0';
		++p;
		lineSize = (int)(p - start);
		int rank = atoi(rankString);
		//printf("%s\n", rankString);
		AddMapInputRecord(spec, &rank, &offset, sizeof(int), sizeof(int));	
		offset += lineSize; 
		start = p;

		if (offset >= fileSize-1) break;
	}
	endTimer("io-test", &loadtimer);

	//------------------------------------------------------
	//main MapReduce procedure
	//------------------------------------------------------
	MapReduce(spec);

	//------------------------------------------------------
	//further processing
	//------------------------------------------------------
#ifdef __OUTPUT__
	validate(spec, 10);
#endif

	//------------------------------------------------------
	//finish
	//------------------------------------------------------
	FinishMapReduce(spec);
	free(h_filebuf);
	endTimer("all-test", &timer);
	return 0;
}
