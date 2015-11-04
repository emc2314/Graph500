#!/bin/sh 

echo "generating data..."
./Gen data 1000000 count

./PageViewCount data

rm data
