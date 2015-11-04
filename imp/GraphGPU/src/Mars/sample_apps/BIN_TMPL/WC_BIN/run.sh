#!/bin/sh

if [ ! -f WordCount ]
then
	echo "the file WordCount doesn't exist"
	exit
fi

for ((i=0; i<4; i++))
do
	cat sample >> data
done

./WordCount data

rm -f data
