#!/bin/sh

#echo "Be sure to compile PageRank2 with appropriate NUM_NODE"
CMDNAME=`basename $0`
if [ $# -ne 2 ]; then
  echo "Usage: $CMDNAME run rwr input_id niteration" 1>&2
  echo "input_id: 1: sample(num_node=16)"
  echo "          2: wiki-Vote(num_node=7115)"
  echo "         14: s14.edge(2.7MB)"
  echo "           ..."
  echo "         22: s22.edge(991MB)"
  exit 1
fi

niteration=$2

if [ $1 -eq 1 ]; then 
  edge="sample" num_node=16
elif [ $1 -eq 2 ]; then
  edge="wiki-Vote.txt" num_node=7115
elif [ $1 -gt 13 -a $1 -lt 23 ]; then
  edge=s${1}.edge num_node=`echo 2^${1} | bc`
fi

query="sample.query"
rwr_in="rwr_in"
rwr_out="rwr_out"
l1norm_out="l1norm_out"
smult_out="smult_out"
new_vector="new_vector"
vector_difference="vector_difference"
diff="difference"
converged="check_convergence"

touch $rwr_in
touch $rwr_out
touch $l1norm_out
touch $smult_out
touch $new_vector
touch $vector_difference
touch $diff
touch $converged
if [ ! -e $edge ]; then
    echo "${edge} doesn't exist, exit"
    exit
fi 

l1norm_result=0
mixing_c=0.85
converge_threshold=0.05

echo "edge: $edge"
echo "num_node=$num_node"
#
echo "initialize"
echo "NormalizeVector"
echo "L1norm"
./L1norm ${query} ${l1norm_out}
echo "ScalarMult"
l1norm_out=`cat ${l1norm_out} | cut -f 2`
./ScalarMult ${query} ${smult_out} ${l1norm_out} ${mixing_c}

echo "Normalization completed"

while [ ${i:=1} -le $niteration ]
do
  echo ""
  echo "ITERATION : $i"
  echo "RWR"
  ./RWR $edge $rwr_in $rwr_out $num_node $mixing_c $i
  echo "SaxpyTextoutput"
  ./Saxpy ${rwr_out} ${smult_out} ${new_vector} 1.0 1 # output is vector
  echo "Saxpy"
  ./Saxpy ${new_vector} ${rwr_in} ${vector_difference} -1.0 0 # output is not vector
  echo "L1norm"
  ./L1norm ${vector_difference} ${diff} $converge_threshold
  if [ `cat ${converged}` ]; then 
      echo "CONVERGED"
      echo "total iteration : ${i}"
      exit
  fi
  mv $new_vector $rwr_in
  i=`expr $i + 1`
done  
