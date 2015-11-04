#!/bin/sh

if [ ! -f StringMatch ]
then
	echo "the file StringMatch doesn't exist"
	exit
fi

for ((i=0; i<1024; i++))
do
	cat sample.txt >> data.txt
done

./StringMatch data.txt org

rm -f data.txt
