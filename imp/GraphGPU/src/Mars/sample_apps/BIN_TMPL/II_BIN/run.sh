#!/bin/sh

if [ ! -f InvertedIndex ]
then
	echo "the file InvertedIndex doesn't exist"
	exit
fi

echo "generating 28 MB data..."
for ((i=0; i<1024; i++))
do
	cat sample/1.html >> data/1.html
	cat sample/2.html >> data/2.html
	cat sample/3.html >> data/3.html
done

./InvertedIndex data/

rm -f data/*
