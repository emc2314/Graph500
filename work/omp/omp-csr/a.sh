if [ $# -eq 6 ]
then
mv csr bakcsr
mv csrtd bakcsrtd
mv csrbu bakcsrbu
sc=$1
while [[ $sc -le $2 ]]
do
a=$3
while [[ $a -le $4 ]]
do
b=$5
while [[ $b -le $6 ]]
do
t=0
r1=0
while [[ $t -lt 3 ]]
do
echo "omp-csr   '$sc' '$a' '$b'"
x=`./omp-csr $sc $a $b | grep harmonic_m | awk '{print $2}' | sed -e 's/[eE]+*/\\*10\\^/'`
x=`echo "scale=5;$x/3"|bc`
r1=`echo "scale=5;$x+$r1"|bc`
t=$(($t+1))
done
echo $r1 | awk '{printf "'$sc' '$a' '$b' %.6e\n", '$r1'}' >> csr
b=$(($b+1))
done
a=$(($a+1))
done
t=0
r2=0
while [[ $t -lt 3 ]]
do
echo "omp-csrtd '$sc'"
x=`./omp-csrtd $sc $a $b | grep harmonic_m | awk '{print $2}' | sed -e 's/[eE]+*/\\*10\\^/'`
x=`echo "scale=5;$x/3"|bc`
r2=`echo "scale=5;$x+$r2"|bc`
t=$(($t+1))
done
echo $r2 | awk '{printf "'$sc' %.6e\n", '$r2'}' >> csrtd
t=0
r3=0
while [[ $t -lt 3 ]]
do
echo "omp-csrbu '$sc'"
x=`./omp-csrbu $sc $a $b | grep harmonic_m | awk '{print $2}' | sed -e 's/[eE]+*/\\*10\\^/'`
x=`echo "scale=5;$x/3"|bc`
r3=`echo "scale=5;$x+$r3"|bc`
t=$(($t+1))
done
echo $r3 | awk '{printf "'$sc' %.6e\n", '$r3'}' >> csrbu
sc=$(($sc+1))
done
else
echo "usage: a.sh SCALE_start SCALE_end ALPHA_start ALPHA_end BETA_start BETA_end"
fi
