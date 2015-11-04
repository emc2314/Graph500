#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
	if (argc != 4)
	{
		printf("usage: %s file recordNum type\n\ttype:\n\t\tcount\n\t\trank\n", argv[0]);
		return -1;
	}
		
	char *fileName = argv[1];
	int recNum = atoi(argv[2]);
	char *type = argv[3];

	srand(time(0));

	if (strcmp(type, "count") == 0)
	{
		FILE *fp = fopen(fileName, "w+");
		
		int i;
		for (i = 0; i < recNum; i++)
			fprintf(fp, "http://www.abcdefg.com/%d.html\t%d.%d.%d.%d\t\%d\n", rand()%1024, rand()%256, rand()%256, rand()%256,rand()%256, rand()%1000);

		fclose(fp);
	}
	else if (strcmp(type, "rank") == 0)
	{
		FILE *fp = fopen(fileName, "w+");
		int i;
		for (i = 0; i < recNum; i++)
			fprintf(fp, "http://www.abcdefg.com/%d.html\t%d\n", rand()%1024, rand());
		fclose(fp);
	}
	else
	{
		printf("usage: %s file recordNum type\n", argv[0]);
		return -1;
	}

	return 0;
}
