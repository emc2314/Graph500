#!/bin/sh

#echo "Be sure to compile PageRank2 with appropriate NUM_NODE"
CMDNAME=`basename $0`
if [ $# -ne 2 ]; then
  echo "Usage: $CMDNAME run pr input_id niteration" 1>&2
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

pr_out="pr_out"
pr_in="pr_in"
converged="converged_reducer"

pr1_out="pr1_out"

touch ${pr_out}
touch ${pr_in}
if [ ! -e $edge ]; then
    echo "${edge} doesn't exist, exit"
    exit
fi 

echo "niter (unless converge) ${niteration}"
while [ ${i:=1} -le ${niteration} ]
  do
  echo "iteration: ${i}"
#  mpirun -np $3 -hostfile machine ./PageRank ${edge} ${pr_in} ${pr_out} ${num_node} ${i}
#  mpirun -np $3 ./PageRank ${edge} ${pr_in} ${pr_out} ${num_node} ${i}
   # ./PageRank1 $edge $pr_in $pr1_out $num_node $i
   # ./PageRank2 $pr1_out $pr_out $num_node
  ./PageRank $edge $pr_in $pr_out $num_node $i

  mv ${pr_out} ${pr_in}
  echo $num_converged
  if [ `cat ${converged}` ]
      then 
      echo "CONVERGED"
      echo "total iteration : ${i}"
      exit
  fi  
  i=`expr ${i} + 1`
done

