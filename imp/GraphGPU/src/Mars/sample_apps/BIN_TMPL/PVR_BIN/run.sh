#!/bin/sh 

echo "generating data..."
./Gen data 1000000 rank

./PageViewRank data

rm data
