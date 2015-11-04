/*$Id: map.cu 730 2009-11-13 13:01:58Z wenbinor $*/
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

#define START		0x00 
#define IN_TAG		0x01 
#define IN_ATAG		0x02 
#define FOUND_HREF	0x03 
#define START_LINK	0x04 

__device__ int hash_func(char* str, int len)
{
	int hash, i;
	for (i = 0, hash=len; i < len; i++)
		hash = (hash<<4)^(hash>>28)^str[i];
	return hash;
}

__device__ void MAP_COUNT_FUNC//(void *key, void *val, size_t keySize, size_t valSize)
{			
	II_KEY_T* pKey = (II_KEY_T*)key;
	II_VAL_T* pVal = (II_VAL_T*)val;

	int block_offset = pVal->block_offset;
	char *buf = pKey->file_buf + block_offset;
	int size = pVal->block_size;
	//printf("%s\n", buf);
	//printf("================\n");
	int state = START;
	char* link_end;
	int j = 0;

	for (j = 0; j < size; j++)
	{
		switch(state)
		{
			case START:
				if (buf[j] == '<') 
				{
					state = IN_TAG;
				//	printf("%c - START -> IN_TAG\n", buf[j]);
				}
				break;
			case IN_TAG:
				if (buf[j] == 'a') 
				{
					state = IN_ATAG;
				//	printf("%c - IN_TAG -> IN_ATAG\n", buf[j]);
				}
				else if (buf[j] == ' ') 
				{
					state = IN_TAG;
				//	printf("%c - IN_TAG -> IN_TAG\n", buf[j]);
				}
				else state = START;
				break;
			case IN_ATAG:
				if (buf[j] == 'h')
				{
					char href[5] = {'h','r','e','f','\0'};
					char* url_start = buf + j;
					
					int x;
					for (x = 0; x < 5; x++)
						if (href[x] != url_start[x]) break;
					if (href[x] == '\0')
					{
						state = FOUND_HREF;
				//		printf("%c - IN_ATAG -> FOUND_HREF, %c\n", buf[j], buf[j+3]);
						j+=3;
					}
					else state = START;
				}
				else if (buf[j] == ' ') state = IN_ATAG;
               			else state = START;
    	       			break;
			case FOUND_HREF:
				if (buf[j] == ' ') state = FOUND_HREF;
				else if (buf[j] == '=') state = FOUND_HREF;
				else if (buf[j] == '\"') 
				{
					state = START_LINK;	
				//	printf("%c - FOUND_HREF -> START_LINK\n", buf[j]);
				}
				else state = START;
				break;
			case START_LINK:
				link_end = NULL;
				link_end = buf + j;
				for (; *link_end != '\"'; link_end++);
				//*link_end = '\0';
				//printf("%s\n", buf + j);
				//emit
				//printf("*link_end:%c\n", *link_end);
				//printf("url:%s\n", buf + j);
				//printf("%c - START_LINK -> START, %c\n", buf[j], buf[(link_end - (buf + j))]);
				//pKey->url_offset = j + block_offset;
				EMIT_INTER_COUNT_FUNC(sizeof(II_KEY_T), sizeof(int));
				j += (link_end - (buf + j));
				state = START;
				break;
		}
	}
}

__device__ void MAP_FUNC//(void *key, void val, size_t keySize, size_t valSize)
{
	II_KEY_T* pKey = (II_KEY_T*)key;
	II_VAL_T* pVal = (II_VAL_T*)val;

	int block_offset = pVal->block_offset;
	char *buf = pKey->file_buf + block_offset;
	int size = pVal->block_size;
	//printf("%s\n", buf);
	//printf("================\n");
	int state = START;
	char* link_end;
	int j = 0;

	for (j = 0; j < size; j++)
	{
		switch(state)
		{
			case START:
				if (buf[j] == '<') 
				{
					state = IN_TAG;
				//	printf("%c - START -> IN_TAG\n", buf[j]);
				}
				break;
			case IN_TAG:
				if (buf[j] == 'a') 
				{
					state = IN_ATAG;
				//	printf("%c - IN_TAG -> IN_ATAG\n", buf[j]);
				}
				else if (buf[j] == ' ') 
				{
					state = IN_TAG;
				//	printf("%c - IN_TAG -> IN_TAG\n", buf[j]);
				}
				else state = START;
				break;
			case IN_ATAG:
				if (buf[j] == 'h')
				{
					char href[5] = {'h','r','e','f','\0'};
					char* url_start = buf + j;
					
					int x;
					for (x = 0; x < 5; x++)
						if (href[x] != url_start[x]) break;
					if (href[x] == '\0')
					{
						state = FOUND_HREF;
				//		printf("%c - IN_ATAG -> FOUND_HREF, %c\n", buf[j], buf[j+3]);
						j+=3;
					}
					else state = START;
				}
				else if (buf[j] == ' ') state = IN_ATAG;
               			else state = START;
    	       			break;
			case FOUND_HREF:
				if (buf[j] == ' ') state = FOUND_HREF;
				else if (buf[j] == '=') state = FOUND_HREF;
				else if (buf[j] == '\"') 
				{
					state = START_LINK;	
				//	printf("%c - FOUND_HREF -> START_LINK\n", buf[j]);
				}
				else state = START;
				break;
			case START_LINK:
				link_end = NULL;
				link_end = buf + j;
				for (; *link_end != '\"'; link_end++);
				*link_end = '\0';
				//printf("%s\n", buf + j);
				//emit
				//printf("*link_end:%c\n", *link_end);
				//printf("url:%s\n", buf + j);
				//printf("%c - START_LINK -> START, %c\n", buf[j], buf[(link_end - (buf + j))]);
				pKey->url_offset = j + block_offset;
#ifdef __HASH__
				pKey->url_hash = hash_func(buf + j, (link_end - (buf + j)));
#endif
				EMIT_INTERMEDIATE_FUNC(key, &pVal->file_id, sizeof(II_KEY_T), sizeof(int));
				j += (link_end - (buf + j));
				state = START;
				break;
		}
	}
}
#endif //__MAP_CU__
