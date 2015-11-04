#!/bin/sh

#echo "Be sure to compile PageRank2 with appropriate NUM_NODE"
CMDNAME=`basename $0`
if [ $# -ne 2 ]; then
  echo "Usage: $CMDNAME run cc input_id niteration" 1>&2
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

curbm="curbm"
cc1_out="cc1_out"
cc2_out="cc2_out"
cc3_out="cc3_out"

converged="check_convergence"

rm $converged
touch $converged
if [ ! -e $edge ]; then
    echo "${edge} doesn't exist, exit"
    exit
fi 

echo "edge: $edge"
echo "num_node=$num_node"

# ConCmpt
while [ ${i:=1} -le $niteration ]
do
  echo ""
  echo "ITERATION : $i"
#   echo "ConCmpt 1"
#   ./ConCmpt1 ${edge} ${curbm} ${cc1_out} ${num_node} $i

#   echo "ConCmpt 2"
#   ./ConCmpt2 $cc1_out $cc2_out $num_node
  echo "ConCmpt"
  ./ConCmpt ${edge} ${curbm} ${cc2_out} ${num_node} $i

  mv $cc2_out $curbm
  echo "ConCmpt 3"
  echo "Hop ${i} :"
  ./ConCmpt3 $curbm $cc3_out $converged

  # check convergence
  if [ `cat ${converged}` ]; then 
      echo "CONVERGED"
      echo "total iteration : ${i}"
      exit
  fi

  i=`expr $i + 1`
done

exit




  echo "RWR 1"
  ./RWR1 ${edge} ${rwr_init_v} ${rwr1_out} ${num_node} $i
  echo "RWR 2"
  ./RWR2 ${rwr1_out} ${rwr2_out} ${num_node} ${mixing_c}
  echo "SaxpyTextoutput"
  ./Saxpy ${rwr2_out} ${smult_out} ${new_vector} 1.0 1 # output is vector
  echo "Saxpy"
  ./Saxpy ${new_vector} ${rwr_init_v} ${vector_difference} -1.0 0 # output is not vector
  echo "L1norm"
  ./L1norm ${vector_difference} ${diff} $converge_threshold
  if [ `cat ${converged}` ]; then 
      echo "CONVERGED"
      echo "total iteration : ${i}"
      exit
  fi
  mv $new_vector $rwr_init_v
  i=`expr $i + 1`
done  
