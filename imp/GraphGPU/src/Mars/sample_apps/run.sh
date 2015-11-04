#!/bin/sh
#$Id: run.sh 740 2009-11-13 16:17:34Z wenbinor $

#===========================================================
#user defined variables
#===========================================================

#-----------------------------------------------------------
# without a slash in the tail of the two paths
#-----------------------------------------------------------

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#you must setup correct $SDK_PATH
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#SDK_PATH="$HOME/CUDA_SDK"
SDK_PATH="$HOME/NVIDIA_GPU_Computing_SDK"

SDK_BIN_PATH="$SDK_PATH/C/bin/linux/release"
# SDK_BIN_PATH="$SDK_PATH/C/bin/linux/debug"
SDK_SRC_PATH="$SDK_PATH/C/sample_apps"

BIN_TMPL_PATH=$SDK_SRC_PATH/BIN_TMPL

SS_BIN_DIR=$SDK_BIN_PATH/SimilarityScore
SM_BIN_DIR=$SDK_BIN_PATH/StringMatch
II_BIN_DIR=$SDK_BIN_PATH/InvertedIndex
PVC_BIN_DIR=$SDK_BIN_PATH/PageViewCount
PVR_BIN_DIR=$SDK_BIN_PATH/PageViewRank
MM_BIN_DIR=$SDK_BIN_PATH/MatrixMul
CPUMM_BIN_DIR=$SDK_BIN_PATH/MatrixMul-cpu
WC_BIN_DIR=$SDK_BIN_PATH/WordCount
KM_BIN_DIR=$SDK_BIN_PATH/Kmeans
CPUKM_BIN_DIR=$SDK_BIN_PATH/Kmeans-cpu
PR_BIN_DIR=$SDK_BIN_PATH/PageRank
MPIPR_BIN_DIR=$SDK_BIN_PATH/PageRank-mpi
CPUPR_BIN_DIR=$SDK_BIN_PATH/PageRank-cpu
CPUMPIPR_BIN_DIR=$SDK_BIN_PATH/PageRank-cpumpi
SP_BIN_DIR=$SDK_BIN_PATH/SimPart
SPMPI_BIN_DIR=$SDK_BIN_PATH/SimPart-mpi
CPUMPIPR1_BIN_DIR=$SDK_BIN/PageRank-cpumpi
PR1_BIN_DIR=$SDK_BIN_PATH/PageRank1
MPIPR1_BIN_DIR=$SDK_BIN_PATH/PageRank1-mpi
CPUPR1_BIN_DIR=$SDK_BIN_PATH/PageRank1-cpu
PR2_BIN_DIR=$SDK_BIN_PATH/PageRank2
MPIPR2_BIN_DIR=$SDK_BIN_PATH/PageRank2-mpi
CPUPR2_BIN_DIR=$SDK_BIN_PATH/PageRank2-cpu
PRD_BIN_DIR=$SDK_BIN_PATH/PageRankDouble
PRD1_BIN_DIR=$SDK_BIN_PATH/PageRankDouble1
PRD2_BIN_DIR=$SDK_BIN_PATH/PageRankDouble2
RWR_BIN_DIR=$SDK_BIN_PATH/RWR
CPURWR_BIN_DIR=$SDK_BIN_PATH/RWR-cpu
RWR1_BIN_DIR=$SDK_BIN_PATH/RWR1
RWR2_BIN_DIR=$SDK_BIN_PATH/RWR2
L1N_BIN_DIR=$SDK_BIN_PATH/L1norm
L2N_BIN_DIR=$SDK_BIN_PATH/L2norm
SQ_BIN_DIR=$SDK_BIN_PATH/Square
SQ2_BIN_DIR=$SDK_BIN_PATH/Square2
SMULT_BIN_DIR=$SDK_BIN_PATH/ScalarMult
SAXPY_BIN_DIR=$SDK_BIN_PATH/Saxpy
MV_BIN_DIR=$SDK_BIN_PATH/MatVecMul
MVW_BIN_DIR=$SDK_BIN_PATH/MatVecMulWeight
LZ_BIN_DIR=$SDK_BIN_PATH/Lanczos
LZS_BIN_DIR=$SDK_BIN_PATH/Lanczos-stream
CC_BIN_DIR=$SDK_BIN_PATH/ConCmpt
MPICC_BIN_DIR=$SDK_BIN_PATH/ConCmpt-mpi
CPUCC_BIN_DIR=$SDK_BIN_PATH/ConCmpt-cpu
CPUMPICC_BIN_DIR=$SDK_BIN_PATH/ConCmpt-cpumpi
CC1_BIN_DIR=$SDK_BIN_PATH/ConCmpt1
CC2_BIN_DIR=$SDK_BIN_PATH/ConCmpt2
CC3_BIN_DIR=$SDK_BIN_PATH/ConCmpt3
AM_BIN_DIR=$SDK_BIN_PATH/Alignment
ST_BIN_DIR=$SDK_BIN_PATH/Sort

SS_SRC_DIR=$SDK_SRC_PATH/SimilarityScore
SM_SRC_DIR=$SDK_SRC_PATH/StringMatch
II_SRC_DIR=$SDK_SRC_PATH/InvertedIndex
PVC_SRC_DIR=$SDK_SRC_PATH/PageViewCount
PVR_SRC_DIR=$SDK_SRC_PATH/PageViewRank
MM_SRC_DIR=$SDK_SRC_PATH/MatrixMul
CPUMM_SRC_DIR=$SDK_SRC_PATH/MatrixMul-cpu
WC_SRC_DIR=$SDK_SRC_PATH/WordCount
KM_SRC_DIR=$SDK_SRC_PATH/Kmeans
CPUKM_SRC_DIR=$SDK_SRC_PATH/Kmeans-cpu
PR_SRC_DIR=$SDK_SRC_PATH/PageRank
CPUPR_SRC_DIR=$SDK_SRC_PATH/PageRank-cpu
MPIPR_SRC_DIR=$SDK_SRC_PATH/PageRank-mpi
CPUMPIPR_SRC_DIR=$SDK_SRC_PATH/PageRank-cpumpi
SP_SRC_DIR=$SDK_SRC_PATH/SimPart
SPMPI_SRC_DIR=$SDK_SRC_PATH/SimPart-mpi
CPUMPIPR1_SRC_DIR=$SDK_SRC_PATH/PageRank1-cpumpi
PR1_SRC_DIR=$SDK_SRC_PATH/PageRank1
MPIPR1_SRC_DIR=$SDK_SRC_PATH/PageRank1-mpi
CPUPR1_SRC_DIR=$SDK_SRC_PATH/PageRank1-cpu
PR2_SRC_DIR=$SDK_SRC_PATH/PageRank2
MPIPR2_SRC_DIR=$SDK_SRC_PATH/PageRank2-mpi
CPUPR2_SRC_DIR=$SDK_SRC_PATH/PageRank2-cpu
PRD1_SRC_DIR=$SDK_SRC_PATH/PageRankDouble1
PRD2_SRC_DIR=$SDK_SRC_PATH/PageRankDouble2
RWR_SRC_DIR=$SDK_SRC_PATH/RWR
CPURWR_SRC_DIR=$SDK_SRC_PATH/RWR-cpu
RWR1_SRC_DIR=$SDK_SRC_PATH/RWR1
RWR2_SRC_DIR=$SDK_SRC_PATH/RWR2
L1N_SRC_DIR=$SDK_SRC_PATH/L1norm
L2N_SRC_DIR=$SDK_SRC_PATH/L2norm
SQ_SRC_DIR=$SDK_SRC_PATH/Square
SQ2_SRC_DIR=$SDK_SRC_PATH/Square2
SMULT_SRC_DIR=$SDK_SRC_PATH/ScalarMult
SAXPY_SRC_DIR=$SDK_SRC_PATH/Saxpy
MV_SRC_DIR=$SDK_SRC_PATH/MatVecMul
MVW_SRC_DIR=$SDK_SRC_PATH/MatVecMulWeight
LZ_SRC_DIR=$SDK_SRC_PATH/Lanczos
LZS_SRC_DIR=$SDK_SRC_PATH/Lanczos-stream
CC_SRC_DIR=$SDK_SRC_PATH/ConCmpt
MPICC_SRC_DIR=$SDK_SRC_PATH/ConCmpt-mpi
CPUCC_SRC_DIR=$SDK_SRC_PATH/ConCmpt-cpu
CPUMPICC_SRC_DIR=$SDK_SRC_PATH/ConCmpt-cpumpi
C1_SRC_DIR=$SDK_SRC_PATH/ConCmpt1
CC2_SRC_DIR=$SDK_SRC_PATH/ConCmpt2
CC3_SRC_DIR=$SDK_SRC_PATH/ConCmpt3
AM_SRC_DIR=$SDK_SRC_PATH/Alignment
ST_SRC_DIR=$SDK_SRC_PATH/Sort

USAGE="usage: run.sh [make|run] [all|ss|sm|mm|pvc|pvr|ii|wc|km|\
       pr|rwr|cc|pr-cpumpi|sp|sp-mpi]"
STR_MAKE="making source code..."
STR_RUN="running test suite..."
STR_MAKE_SS="making Similarity Score source code..."
STR_MAKE_SM="making String Match source code..."
STR_MAKE_MM="making MatrixMul source code..."
STR_MAKE_PVC="making PageViewCount source code..."
STR_MAKE_PVR="making PageViewRank source code..."
STR_MAKE_ALL="making all source code..."
STR_MAKE_II="making InvertedIndex source code..."
STR_MAKE_WC="making WordCount source code..."
STR_MAKE_KM="making Kmeans source code..."
STR_MAKE_SP="making SimPart source code..."
STR_MAKE_SPMPI="making SimPart-mpi source code..."
STR_MAKE_PR="making PageRank source code..."
STR_MAKE_PR1="making PageRank1 source code..."
STR_MAKE_MPIPR="making PageRank-mpi source code..."
STR_MAKE_MPIPR1="making PageRank1-mpi source code..."
STR_MAKE_MPIPR2="making PageRank2-mpi source code..."
STR_MAKE_CPUPR="making PageRank-cpu source code..."
STR_MAKE_CPUPR1="making PageRank1-cpu source code..."
STR_MAKE_CPUPR2="making PageRank2-cpu source code..."
STR_MAKE_CPUMPIPR="making PageRank-cpumpi source code..."
STR_MAKE_CPUMPIPR1="making PageRank1-cpumpi source code..."
STR_MAKE_PR2="making PageRank2 source code..."
STR_MAKE_PRD="making PageRankDouble source code..."
STR_MAKE_PRD1="making PageRankDouble1 source code..."
STR_MAKE_PRD2="making PageRankDouble2 source code..."
STR_MAKE_RWR="making RWR source code..."
STR_MAKE_RWR1="making RWR1 source code..."
STR_MAKE_RWR2="making RWR2 source code..."
STR_MAKE_L1N="making L1norm source code..."
STR_MAKE_L2N="making L2norm source code..."
STR_MAKE_SQ="making Square source code..."
STR_MAKE_SQ2="making Square2 source code..."
STR_MAKE_SMULT="making ScalarMult source code..."
STR_MAKE_SAXPY="making Saxpy source code..."
STR_MAKE_MV="making MatVecMul source code..."
STR_MAKE_MVW="making MatVecMulWeight source code..."
STR_MAKE_LZ="making Lanczos source code..."
STR_MAKE_LZS="making Lanczos-stream source code..."
STR_MAKE_CC="making ConCmpt source code..."
STR_MAKE_MPICC="making ConCmpt-mpi source code..."
STR_MAKE_CPUCC="making ConCmpt-cpu source code..."
STR_MAKE_CPUMPICC="making ConCmpt-cpumpi source code..."
STR_MAKE_CC1="making ConCmpt1 source code..."
STR_MAKE_CC2="making ConCmpt2 source code..."
STR_MAKE_CC3="making ConCmpt3 source code..."
STR_MAKE_AM="making Alignment source code..."
STR_MAKE_ST="making Sort source code..."

#===========================================================
#user defined functions
#===========================================================

function MakeSrc()
{
	if [ ! -d $1 ]
	then
		echo "$1 doesn't exist, mkdir it..."
		mkdir $1
	fi

	echo "enter $2, making..."
	cd $2
	sh make.sh $3
#	echo "made successfully"
}

#check the number of command line arguments
if [ $# -lt 2 ]
then
	echo $USAGE
	exit
fi

#run the test suite
if [ "$1" = "run" ]
then
	echo $STR_RUN

	case $2 in
		"ss")
			cp -r $BIN_TMPL_PATH/SS_BIN/* $SS_BIN_DIR/
			cd $SS_BIN_DIR
			sh run.sh
			;;
		"sm")
			cp -r $BIN_TMPL_PATH/SM_BIN/* $SM_BIN_DIR/
			cd $SM_BIN_DIR
			sh run.sh
			;;
		"mm")
			cp -r $BIN_TMPL_PATH/MM_BIN/* $MM_BIN_DIR/
			cd $MM_BIN_DIR
			sh run.sh
			;;
		"mm-cpu")
                        if [ ! -d $CPUMM_BIN_DIR ] 
                            then
                            echo "$CPUMM_BIN_DIR doesn't exist, mkdir it..."
                            mkdir $CPUMM_BIN_DIR
			fi
			cp -r $BIN_TMPL_PATH/CPUMM_BIN/* $CPUMM_BIN_DIR/
			cd $CPUMM_BIN_DIR
			sh run.sh
			;;
		"ii")
			cp -r $BIN_TMPL_PATH/II_BIN/* $II_BIN_DIR/
			cd $II_BIN_DIR
			sh run.sh
			;;
		"pvc")
			cd $BIN_TMPL_PATH/GenWebLogSrc
			make clean 
			make
			cp Gen ../PVC_BIN
			cp ../PVC_BIN/* $PVC_BIN_DIR/
			chmod 777 $PVC_BIN_DIR/Gen
			cd $PVC_BIN_DIR
			sh run.sh
			;;
		"pvr")
			cd $BIN_TMPL_PATH/GenWebLogSrc
			make clean
			make
			cp Gen ../PVR_BIN
			cp ../PVR_BIN/* $PVR_BIN_DIR/
			chmod 777 $PVR_BIN_DIR/Gen
			cd $PVR_BIN_DIR
			sh run.sh
			;;
		"wc")
			cp -r $BIN_TMPL_PATH/WC_BIN/* $WC_BIN_DIR/
			cd $WC_BIN_DIR
			sh run.sh
			;;
		"km")
			cp -r $BIN_TMPL_PATH/KM_BIN/* $KM_BIN_DIR/
			cd $KM_BIN_DIR
			sh run.sh
			;;
		"km-cpu")
                        if [ ! -d $CPUKM_BIN_DIR ] 
                            then
                            echo "$CPUKM_BIN_DIR doesn't exist, mkdir it..."
                            mkdir $CPUKM_BIN_DIR
			fi
			cp -r $BIN_TMPL_PATH/CPUKM_BIN/* $CPUKM_BIN_DIR/
			cd $CPUKM_BIN_DIR
			sh run.sh
			;;

		# "pr")
		#         if [ ! -d $PR_BIN_DIR ] 
		# 	    then
		# 	    echo "$PR_BIN_DIR doesn't exist, mkdir it..."
		# 	    mkdir $PR_BIN_DIR
		# 	fi
		# 	cp -r $BIN_TMPL_PATH/PR_BIN/* $PR_BIN_DIR/
		# 	cp $PR1_BIN_DIR/PageRank2 $PR_BIN_DIR
		# 	cp $PR2_BIN_DIR/PageRank2 $PR_BIN_DIR
		# 	cd $PR_BIN_DIR
		# 	sh run.sh $3 $4
		# 	;;
		"pr")
		        if [ ! -d $PR_BIN_DIR ] 
			    then
			    echo "$PR_BIN_DIR doesn't exist, mkdir it..."
			    mkdir $PR_BIN_DIR
			fi
			cp -r $BIN_TMPL_PATH/PR_BIN/* $PR_BIN_DIR/
			cd $PR_BIN_DIR
			sh run.sh $3 $4
			;;
 		"pr-mpi")
 		        if [ ! -d $MPIPR_BIN_DIR ] 
 			    then
 			    echo "$MPIPR_BIN_DIR doesn't exist, mkdir it..."
 			    mkdir $MPIPR_BIN_DIR
 			fi
 			cp -r $BIN_TMPL_PATH/MPIPR_BIN/* $MPIPR_BIN_DIR/
 			# cp $MPIPR1_BIN_DIR/PageRank1-mpi $MPIPR_BIN_DIR
 			# cp $MPIPR2_BIN_DIR/PageRank2-mpi $MPIPR_BIN_DIR
 			cd $MPIPR_BIN_DIR
 			sh run.sh $3 $4 $5
 			;;
                "pr-cpu")
                        if [ ! -d $CPUPR_BIN_DIR ] 
                            then
                            echo "$CPUPR_BIN_DIR doesn't exist, mkdir it..."
                            mkdir $CPUPR_BIN_DIR
                        fi
                        cp -r $BIN_TMPL_PATH/CPUPR_BIN/* $CPUPR_BIN_DIR/
                        # cp $CPUPR1_BIN_DIR/PageRank1-cpu $CPUPR_BIN_DIR
                        # cp $CPUPR2_BIN_DIR/PageRank2-cpu $CPUPR_BIN_DIR
                        cd $CPUPR_BIN_DIR
                        sh run.sh $3 $4 $5 # $3: inputID, $4: niter, $5: ncpus
                        ;;
                "pr-cpumpi")
                        if [ ! -d $CPUMPIPR_BIN_DIR ] 
                            then
                            echo "$CPUMPIPR_BIN_DIR doesn't exist, mkdir it..."
                            mkdir $CPUMPIPR_BIN_DIR
                        fi
                        cp -r $BIN_TMPL_PATH/CPUMPIPR_BIN/* $CPUMPIPR_BIN_DIR/
                        cd $CPUMPIPR_BIN_DIR
                        # $3: inputID, $4: nprocs, $5: ncpus, $6: ngpus $7: niter $8:$partmethod
                        sh run.sh $3 $4 $5 $6 $7 $8
                        ;;
                "sp")
                        if [ ! -d $SP_BIN_DIR ] 
                            then
                            echo "$SP_BIN_DIR doesn't exist, mkdir it..."
                            mkdir $SP_BIN_DIR
                        fi
                        cp -r $BIN_TMPL_PATH/SP_BIN/* $SP_BIN_DIR/
                        cd $SP_BIN_DIR
                        # $3: inputID, $4: nprocs, $5: ncpus, $6: ngpus $7: niter
                        sh run.sh $3 $4 $5 $6 $7 
                        ;;
                "sp-mpi")
                        if [ ! -d $SPMPI_BIN_DIR ] 
                            then
                            echo "$SPMPI_BIN_DIR doesn't exist, mkdir it..."
                            mkdir $SPMPI_BIN_DIR
                        fi
                        cp -r $BIN_TMPL_PATH/MPISP_BIN/* $SPMPI_BIN_DIR/
                        cd $SPMPI_BIN_DIR
                        # $3: inputID, $4: nprocs, $5: ncpus, $6: ngpus $7: partmethod $8: nparts
                        sh run.sh $3 $4 $5 $6 $7 $8
                        ;;
                "pr1-cpumpi")
                        if [ ! -d $CPUMPIPR1_BIN_DIR ] 
                            then
                            echo "$CPUMPIPR1_BIN_DIR doesn't exist, mkdir it..."
                            mkdir $CPUMPIPR1_BIN_DIR
                        fi
                        cp -r $BIN_TMPL_PATH/CPUMPIPR1_BIN/* $CPUMPIPR1_BIN_DIR/
                        cd $CPUMPIPR1_BIN_DIR
                        sh run.sh $3 $4 $5 $6 # $3: inputID, $4: nprocs, $5: ncpus, $6: niter
                        ;;
		"pr1")
			cp -r $BIN_TMPL_PATH/PR1_BIN/* $PR1_BIN_DIR/
			cd $PR1_BIN_DIR
			sh run.sh $3
			;;
		"pr1-mpi")
			cp -r $BIN_TMPL_PATH/MPIPR1_BIN/* $MPIPR1_BIN_DIR/
			cd $MPIPR1_BIN_DIR
			sh run.sh $3 $4
			;;
                "pr1-cpu")
                        cp -r $BIN_TMPL_PATH/CPUPR1_BIN/* $CPUPR1_BIN_DIR/
                        cd $CPUPR1_BIN_DIR
                        sh run.sh $3 $4
                        ;;
		"pr2")
			cp -r $BIN_TMPL_PATH/PR2_BIN/* $PR2_BIN_DIR/
			cd $PR2_BIN_DIR
			sh run.sh $3 $4
			;;
		"pr2-mpi")
			cp -r $BIN_TMPL_PATH/MPIPR2_BIN/* $MPIPR2_BIN_DIR/
			cd $MPIPR2_BIN_DIR
			sh run.sh $3 $4
			;;
                "pr2-cpu")
                        cp -r $BIN_TMPL_PATH/CPUPR2_BIN/* $CPUPR2_BIN_DIR/
                        cd $CPUPR2_BIN_DIR
                        sh run.sh $3 $4
                        ;;
		"prd")
			cp -r $BIN_TMPL_PATH/PRD_BIN/* $PRD_BIN_DIR/
			cp $PRD1_BIN_DIR/PageRankDouble1 $PRD_BIN_DIR
			cp $PRD2_BIN_DIR/PageRankDouble2 $PRD_BIN_DIR
			cd $PRD_BIN_DIR
			sh run.sh
			;;
		"prd1")
			cp -r $BIN_TMPL_PATH/PRD1_BIN/* $PRD1_BIN_DIR/
			cd $PRD1_BIN_DIR
			sh run.sh
			;;
		"prd2")
			cp -r $BIN_TMPL_PATH/PRD2_BIN/* $PRD2_BIN_DIR/
			cd $PRD2_BIN_DIR
			sh run.sh
			;;
# 		"rwr")
# 		        if [ ! -d $RWR_BIN_DIR ] 
# 			    then
# 			    echo "$RWR_BIN_DIR doesn't exist, mkdir it..."
# 			    mkdir $RWR_BIN_DIR
# 			fi
# 			cp -r $BIN_TMPL_PATH/RWR_BIN/* $RWR_BIN_DIR/
# 			cp $L1N_BIN_DIR/L1norm $RWR_BIN_DIR/
# 			cp $SMULT_BIN_DIR/ScalarMult $RWR_BIN_DIR/
# 			cp $RWR1_BIN_DIR/RWR1 $RWR_BIN_DIR/
# 			cp $RWR2_BIN_DIR/RWR2 $RWR_BIN_DIR/
# 			cp $SAXPY_BIN_DIR/Saxpy $RWR_BIN_DIR/
# 			cd $RWR_BIN_DIR
# 			sh run.sh $3 $4
# 			;;
		"rwr")
		        if [ ! -d $RWR_BIN_DIR ] 
			    then
			    echo "$RWR_BIN_DIR doesn't exist, mkdir it..."
			    mkdir $RWR_BIN_DIR
			fi
			cp -r $BIN_TMPL_PATH/RWR_BIN/* $RWR_BIN_DIR/
			cp $L1N_BIN_DIR/L1norm $RWR_BIN_DIR/
			cp $SMULT_BIN_DIR/ScalarMult $RWR_BIN_DIR/
			cp $SAXPY_BIN_DIR/Saxpy $RWR_BIN_DIR/
			cd $RWR_BIN_DIR
			sh run.sh $3 $4
			;;
		# "rwr-cpu")
		# 	cp -r $BIN_TMPL_PATH/CPURWR_BIN/* $CPURWR_BIN_DIR/
		# 	cd $CPURWR_BIN_DIR
		# 	sh run.sh
		# 	;;
                "rwr-cpu")
                        if [ ! -d $CPURWR_BIN_DIR ] 
                            then
                            echo "$CPURWR_BIN_DIR doesn't exist, mkdir it..."
                            mkdir $CPURWR_BIN_DIR
                        fi
                        cp -r $BIN_TMPL_PATH/CPURWR_BIN/* $CPURWR_BIN_DIR/
                        # cp $CPURWR1_BIN_DIR/PageRank1-cpu $CPURWR_BIN_DIR
                        # cp $CPURWR2_BIN_DIR/PageRank2-cpu $CPURWR_BIN_DIR
                        cd $CPURWR_BIN_DIR
                        sh run.sh $3 $4 $5 # $3: inputID, $4: niter, $5: ncpus
                        ;;
		"rwr1")
			cp -r $BIN_TMPL_PATH/RWR1_BIN/* $RWR1_BIN_DIR/
			cd $RWR1_BIN_DIR
			sh run.sh
			;;
		"rwr2")
			cp -r $BIN_TMPL_PATH/RWR2_BIN/* $RWR2_BIN_DIR/
			cd $RWR2_BIN_DIR
			sh run.sh
			;;
		"l1n")
			cp -r $BIN_TMPL_PATH/L1N_BIN/* $L1N_BIN_DIR/
			cd $L1N_BIN_DIR
			sh run.sh
			;;
		"l2n")
			cp -r $BIN_TMPL_PATH/L2N_BIN/* $L2N_BIN_DIR/
			cd $L2N_BIN_DIR
			sh run.sh
			;;
		"sq")
			cp -r $BIN_TMPL_PATH/SQ_BIN/* $SQ_BIN_DIR/
			cd $SQ_BIN_DIR
			sh run.sh
			;;
		"sq2")
			cp -r $BIN_TMPL_PATH/SQ2_BIN/* $SQ2_BIN_DIR/
			cd $SQ2_BIN_DIR
			sh run.sh $3 $4
			;;
		"smult")
			cp -r $BIN_TMPL_PATH/SMULT_BIN/* $SMULT_BIN_DIR/
			cd $SMULT_BIN_DIR
			sh run.sh $3
			;;
		"saxpy")
		        if [ ! -d $SAXPY_BIN_DIR ] 
			    then
			    echo "$SAXPY_BIN_DIR doesn't exist, mkdir it..."
			    mkdir $SAXPY_BIN_DIR
			fi
			cp -r $BIN_TMPL_PATH/SAXPY_BIN/* $SAXPY_BIN_DIR/
			cd $SAXPY_BIN_DIR
			sh run.sh $3 $4 $5
			;;
		"mv")
		        if [ ! -d $MV_BIN_DIR ] 
			    then
			    echo "$MV_BIN_DIR doesn't exist, mkdir it..."
			    mkdir $MV_BIN_DIR
			fi
			cp -r $BIN_TMPL_PATH/MV_BIN/* $MV_BIN_DIR/
			cd $MV_BIN_DIR
			sh run.sh $3 $4 
			;;
		"mvw")
		        if [ ! -d $MVW_BIN_DIR ] 
			    then
			    echo "$MVW_BIN_DIR doesn't exist, mkdir it..."
			    mkdir $MVW_BIN_DIR
			fi
			cp -r $BIN_TMPL_PATH/MVW_BIN/* $MVW_BIN_DIR/
			cd $MVW_BIN_DIR
			sh run.sh $3 $4 
			;;
		"lz")
		        if [ ! -d $LZ_BIN_DIR ] 
			    then
			    echo "$LZ_BIN_DIR doesn't exist, mkdir it..."
			    mkdir $LZ_BIN_DIR
			fi
			cp -r $BIN_TMPL_PATH/LZ_BIN/* $LZ_BIN_DIR/
			cd $LZ_BIN_DIR
			sh run.sh $3 $4 $5
			;;
		"lzs")
		        if [ ! -d $LZS_BIN_DIR ] 
			    then
			    echo "$LZS_BIN_DIR doesn't exist, mkdir it..."
			    mkdir $LZS_BIN_DIR
			fi
			cp -r $BIN_TMPL_PATH/LZS_BIN/* $LZS_BIN_DIR/
			cd $LZS_BIN_DIR
			sh run.sh $3 $4 $5 $6
			;;
		"cc")
		        if [ ! -d $CC_BIN_DIR ] 
			    then
			    echo "$CC_BIN_DIR doesn't exist, mkdir it..."
			    mkdir $CC_BIN_DIR
			fi
			cp -r $BIN_TMPL_PATH/CC_BIN/* $CC_BIN_DIR/
 			cp $CC3_BIN_DIR/ConCmpt3 $CC_BIN_DIR/
			cd $CC_BIN_DIR
			sh run.sh $3 $4
			;;
		"cc-mpi")
		        if [ ! -d $MPICC_BIN_DIR ] 
			    then
			    echo "$MPICC_BIN_DIR doesn't exist, mkdir it..."
			    mkdir $MPICC_BIN_DIR
			fi
			cp -r $BIN_TMPL_PATH/MPICC_BIN/* $MPICC_BIN_DIR/
			cd $MPICC_BIN_DIR
			sh run.sh $3 $4 $5
			;;
		"cc-cpu")
		        if [ ! -d $CPUCC_BIN_DIR ] 
			    then
			    echo "$CPUCC_BIN_DIR doesn't exist, mkdir it..."
			    mkdir $CPUCC_BIN_DIR
			fi
			cp -r $BIN_TMPL_PATH/CPUCC_BIN/* $CPUCC_BIN_DIR/
 			# cp $CC3_BIN_DIR/ConCmpt3 $CC_BIN_DIR/
			cd $CPUCC_BIN_DIR
			sh run.sh $3 $4 $5
			;;
		"cc-cpumpi")
		        if [ ! -d $CPUMPICC_BIN_DIR ] 
			    then
			    echo "$CPUMPICC_BIN_DIR doesn't exist, mkdir it..."
			    mkdir $CPUMPICC_BIN_DIR
			fi
			cp -r $BIN_TMPL_PATH/CPUMPICC_BIN/* $CPUMPICC_BIN_DIR/
 			# cp $CC3_BIN_DIR/ConCmpt3 $CC_BIN_DIR/
			cd $CPUMPICC_BIN_DIR
			sh run.sh $3 $4 $5 $6
			;;
# 		"cc")
# 		        if [ ! -d $CC_BIN_DIR ] 
# 			    then
# 			    echo "$CC_BIN_DIR doesn't exist, mkdir it..."
# 			    mkdir $CC_BIN_DIR
# 			fi
# 			cp -r $BIN_TMPL_PATH/CC_BIN/* $CC_BIN_DIR/
# 			cp $CC1_BIN_DIR/ConCmpt1 $CC_BIN_DIR/
# 			cp $CC2_BIN_DIR/ConCmpt2 $CC_BIN_DIR/
# 			cp $CC3_BIN_DIR/ConCmpt3 $CC_BIN_DIR/
# 			cd $CC_BIN_DIR
# 			sh run.sh $3 $4
# 			;;
		"cc1")
			cp -r $BIN_TMPL_PATH/CC1_BIN/* $CC1_BIN_DIR/
			cd $CC1_BIN_DIR
			sh run.sh
			;;
		"cc2")
			cp -r $BIN_TMPL_PATH/CC2_BIN/* $CC2_BIN_DIR/
			cd $CC2_BIN_DIR
			sh run.sh
			;;
		"cc3")
			cp -r $BIN_TMPL_PATH/CC3_BIN/* $CC3_BIN_DIR/
			cd $CC3_BIN_DIR
			sh run.sh
			;;
		"am")
			cp -r $BIN_TMPL_PATH/AM_BIN/* $AM_BIN_DIR/
			cd $AM_BIN_DIR
			sh run.sh
			;;
		"st")
			cp -r $BIN_TMPL_PATH/ST_BIN/* $ST_BIN_DIR/
			cd $ST_BIN_DIR
			sh run.sh $3 $4 $5
			;;
		"all")
			echo "==========Similarity Score========="
			cp -r $BIN_TMPL_PATH/SS_BIN/* $SS_BIN_DIR/
			cd $SS_BIN_DIR
			sh run.sh

			echo "==========StringMatch========="
			cp -r $BIN_TMPL_PATH/SM_BIN/* $SM_BIN_DIR/
			cd $SM_BIN_DIR
			sh run.sh

			echo "==========MatrixMul========="
			cp -r $BIN_TMPL_PATH/MM_BIN/* $MM_BIN_DIR/
			cd $MM_BIN_DIR
			sh run.sh

			echo "==========InvertdIndex========="
			cp -r $BIN_TMPL_PATH/II_BIN/* $II_BIN_DIR/
			cd $II_BIN_DIR
			sh run.sh

			echo "==========PageViewCount========="
			cd  $BIN_TMPL_PATH/GenWebLogSrc
			make clean
			make
			cp Gen ../PVC_BIN
			cp ../PVC_BIN/* $PVC_BIN_DIR/
			chmod 777 $PVC_BIN_DIR/Gen
			cd $PVC_BIN_DIR
			sh run.sh

			echo "==========PageViewRank========="
			cd  $BIN_TMPL_PATH/GenWebLogSrc
			make clean
			make
			cp Gen ../PVR_BIN
			cp ../PVR_BIN/* $PVR_BIN_DIR/
			chmod 777 $PVR_BIN_DIR/Gen
			cd $PVR_BIN_DIR
			sh run.sh

			echo "==========WordCount========="
			cp -r $BIN_TMPL_PATH/WC_BIN/* $WC_BIN_DIR/
			cd $WC_BIN_DIR
			sh run.sh

			echo "==========Kmeans========="
			cp -r $BIN_TMPL_PATH/KM_BIN/* $KM_BIN_DIR/
			cd $KM_BIN_DIR
			sh run.sh

			;;
		*)
			echo $USAGE
			exit
			;;
	esac
#make source code
elif [ "$1" = "make" ]
then
	echo $STR_MAKE

	case $2 in
		"ss")
			echo $STR_MAKE_SS
			MakeSrc $SS_BIN_DIR $SS_SRC_DIR
			;;
		"sm")
			echo $STR_MAKE_SM
			MakeSrc $SM_BIN_DIR $SM_SRC_DIR
			;;
		"mm")
			echo $STR_MAKE_MM
			MakeSrc $MM_BIN_DIR $MM_SRC_DIR
			;;
		"mm-cpu")
			echo $STR_MAKE_MM
			MakeSrc $CPUMM_BIN_DIR $CPUMM_SRC_DIR
			;;
		"pvc")
			echo $STR_MAKE_SM
			MakeSrc $PVC_BIN_DIR $PVC_SRC_DIR
			;;
		"pvr")
			echo $STR_MAKE_SM
			MakeSrc $PVR_BIN_DIR $PVR_SRC_DIR
			;;
		"ii")
			echo $STR_MAKE_SM
			MakeSrc $II_BIN_DIR $II_SRC_DIR
			;;
		"wc")
			echo $STR_MAKE_WC
			MakeSrc $WC_BIN_DIR $WC_SRC_DIR
			;;
		"km")
			echo $STR_MAKE_KM
			MakeSrc $KM_BIN_DIR $KM_SRC_DIR
			;;
		"km-cpu")
			echo $STR_MAKE_KM
			MakeSrc $CPUKM_BIN_DIR $CPUKM_SRC_DIR
			;;
	        "pr")
			echo $STR_MAKE_PR
			MakeSrc $PR_BIN_DIR $PR_SRC_DIR
			;;
	        # "pr")
		# 	echo $STR_MAKE_PR1
		# 	MakeSrc $PR1_BIN_DIR $PR1_SRC_DIR
  		# 	echo $STR_MAKE_PR2
		# 	MakeSrc $PR2_BIN_DIR $PR2_SRC_DIR $3
		# 	;;
                "pr-cpu")
                        # echo $STR_MAKE_CPUPR1
                        # MakeSrc $CPUPR1_BIN_DIR $CPUPR1_SRC_DIR
                        # echo $STR_MAKE_CPUPR2
                        # MakeSrc $CPUPR1_BIN_DIR $CPUPR2_SRC_DIR $3
                        # ;;
			echo $STR_MAKE_CPUPR
			MakeSrc $CPUPR_BIN_DIR $CPUPR_SRC_DIR
 			;;
 	        "pr-mpi")
 			echo $STR_MAKE_MPIPR
			MakeSrc $MPIPR_BIN_DIR $MPIPR_SRC_DIR
 			;;

 			# echo $STR_MAKE_PR1
			# MakeSrc $MPIPR1_BIN_DIR $MPIPR1_SRC_DIR
   			# echo $STR_MAKE_PR2
 			# MakeSrc $MPIPR2_BIN_DIR $MPIPR2_SRC_DIR $3
 			# ;;
                "pr-cpumpi")
			echo $STR_MAKE_CPUMPIPR
			MakeSrc $CPUMPIPR_BIN_DIR $CPUMPIPR_SRC_DIR
 			;;
                "sp")
			echo $STR_MAKE_SP
			MakeSrc $SP_BIN_DIR $SP_SRC_DIR
 			;;
                "sp-mpi")
			echo $STR_MAKE_SPMPI
			MakeSrc $SPMPI_BIN_DIR $SPMPI_SRC_DIR
 			;;
                "pr1-cpumpi")
			echo $STR_MAKE_CPUMPIPR1
			MakeSrc $CPUMPIPR1_BIN_DIR $CPUMPIPR1_SRC_DIR
 			;;
	        "pr1")
			echo $STR_MAKE_PR1
			MakeSrc $PR1_BIN_DIR $PR1_SRC_DIR
			;;
	        "pr1-mpi")
			echo $STR_MAKE_PR1
			MakeSrc $MPIPR1_BIN_DIR $MPIPR1_SRC_DIR
			;;
                "pr1-cpu")
                        echo $STR_MAKE_CPUPR1
                        MakeSrc $CPUPR1_BIN_DIR $CPUPR1_SRC_DIR
                        ;;
	        "pr2")
			echo $STR_MAKE_PR2
			MakeSrc $PR2_BIN_DIR $PR2_SRC_DIR $3
			;;
	        "pr2-mpi")
			echo $STR_MAKE_PR2
			MakeSrc $MPIPR2_BIN_DIR $MPIPR2_SRC_DIR $3
			;;
                "pr2-cpu")
                        echo $STR_MAKE_CPUPR2
                        MakeSrc $CPUPR2_BIN_DIR $CPUPR2_SRC_DIR $3
                        ;;
	        "prd")
			echo $STR_MAKE_PRD1
			MakeSrc $PRD1_BIN_DIR $PRD1_SRC_DIR
  			echo $STR_MAKE_PRD2
			MakeSrc $PRD1_BIN_DIR $PRD2_SRC_DIR
			;;
	        "prd1")
			echo $STR_MAKE_PRD1
			MakeSrc $PRD1_BIN_DIR $PRD1_SRC_DIR
			;;
	        "prd2")
			echo $STR_MAKE_PRD2
			MakeSrc $PRD2_BIN_DIR $PRD2_SRC_DIR
			;;
	        # "rwr")
		# 	echo $STR_MAKE_RWR
		# 	MakeSrc $RWR_BIN_DIR $RWR_SRC_DIR
		# 	;;
	        "rwr-cpu")
			echo $STR_MAKE_RWR
			MakeSrc $CPURWR_BIN_DIR $CPURWR_SRC_DIR
			;;
	        "rwr")
			echo $STR_MAKE_L1N
			MakeSrc $L1N_BIN_DIR $L1N_SRC_DIR
			echo $STR_MAKE_SMULT
			MakeSrc $SMULT_BIN_DIR $SMULT_SRC_DIR
			echo $STR_MAKE_RWR
			MakeSrc $RWR_BIN_DIR $RWR_SRC_DIR
			# echo $STR_MAKE_RWR1
			# MakeSrc $RWR1_BIN_DIR $RWR1_SRC_DIR
			# echo $STR_MAKE_RWR2
			# MakeSrc $RWR2_BIN_DIR $RWR2_SRC_DIR
			echo $STR_MAKE_SAXPY
			MakeSrc $SAXPY_BIN_DIR $SAXPY_SRC_DIR
			;;
	        "rwr1")
			echo $STR_MAKE_RWR1
			MakeSrc $RWR1_BIN_DIR $RWR1_SRC_DIR
			;;
	        "rwr2")
			echo $STR_MAKE_RWR2
			MakeSrc $RWR2_BIN_DIR $RWR2_SRC_DIR
			;;
	        "l1n")
			echo $STR_MAKE_L1N
			MakeSrc $L1N_BIN_DIR $L1N_SRC_DIR
			;;
	        "l2n")
			echo $STR_MAKE_L2N
			MakeSrc $L2N_BIN_DIR $L2N_SRC_DIR
			;;
	        "sq")
			echo $STR_MAKE_SQ
			MakeSrc $SQ_BIN_DIR $SQ_SRC_DIR
			;;
	        "sq2")
			echo $STR_MAKE_SQ2
			MakeSrc $SQ2_BIN_DIR $SQ2_SRC_DIR
			;;
	        "smult")
			echo $STR_MAKE_SMULT
			MakeSrc $SMULT_BIN_DIR $SMULT_SRC_DIR
			;;
	        "saxpy")
			echo $STR_MAKE_SAXPY
			MakeSrc $SAXPY_BIN_DIR $SAXPY_SRC_DIR
			;;
	        "mv")
			echo $STR_MAKE_MV
			MakeSrc $MV_BIN_DIR $MV_SRC_DIR
			;;
	        "mvw")
			echo $STR_MAKE_MVW
			MakeSrc $MVW_BIN_DIR $MVW_SRC_DIR
			;;
	        "lz")
			echo $STR_MAKE_LZ
			MakeSrc $LZ_BIN_DIR $LZ_SRC_DIR
			;;
	        "lzs")
			echo $STR_MAKE_LZS
			MakeSrc $LZS_BIN_DIR $LZS_SRC_DIR
			;;
	        "cc")
			echo $STR_MAKE_CC
			MakeSrc $CC_BIN_DIR $CC_SRC_DIR
			echo $STR_MAKE_CC3
			MakeSrc $CC3_BIN_DIR $CC3_SRC_DIR
			;;
	        "cc-mpi")
			echo $STR_MAKE_CC
			MakeSrc $MPICC_BIN_DIR $MPICC_SRC_DIR
			;;
	        "cc-cpu")
			echo $STR_MAKE_CC
			MakeSrc $CPUCC_BIN_DIR $CPUCC_SRC_DIR
			;;
	        "cc-cpumpi")
			echo $STR_MAKE_CPUMPICC
			MakeSrc $CPUMPICC_BIN_DIR $CPUMPICC_SRC_DIR
			;;
			# echo $STR_MAKE_CC3
			# MakeSrc $CC3_BIN_DIR $CC3_SRC_DIR
# 	        "cc")
# 			echo $STR_MAKE_CC1
# 			MakeSrc $CC1_BIN_DIR $CC1_SRC_DIR
# 			echo $STR_MAKE_CC2
# 			MakeSrc $CC2_BIN_DIR $CC2_SRC_DIR
# 			echo $STR_MAKE_CC3
# 			MakeSrc $CC3_BIN_DIR $CC3_SRC_DIR
# 			;;
	        "cc1")
			echo $STR_MAKE_CC1
			MakeSrc $CC1_BIN_DIR $CC1_SRC_DIR
			;;
	        "cc2")
			echo $STR_MAKE_CC2
			MakeSrc $CC2_BIN_DIR $CC2_SRC_DIR
			;;
	        "cc3")
			echo $STR_MAKE_CC3
			MakeSrc $CC3_BIN_DIR $CC3_SRC_DIR
			;;
	        "am")
			echo $STR_MAKE_AM
			MakeSrc $AM_BIN_DIR $AM_SRC_DIR
			;;
	        "st")
			echo $STR_MAKE_ST
			MakeSrc $ST_BIN_DIR $ST_SRC_DIR
			;;
		"all")
			echo $STR_MAKE_ALL
			MakeSrc $SS_BIN_DIR $SS_SRC_DIR
			MakeSrc $SM_BIN_DIR $SM_SRC_DIR
			MakeSrc $MM_BIN_DIR $MM_SRC_DIR
			MakeSrc $PVC_BIN_DIR $PVC_SRC_DIR
			MakeSrc $PVR_BIN_DIR $PVR_SRC_DIR
			MakeSrc $II_BIN_DIR $II_SRC_DIR
			MakeSrc $WC_BIN_DIR $WC_SRC_DIR
			MakeSrc $KM_BIN_DIR $KM_SRC_DIR
			echo "all done"
			;;
		*)
			echo $USAGE
			exit
			;;
	esac

#clean object files
elif [ "$1" = "clean" ]
then
	echo "clean obj files..."
	case $2 in
		"ss")
			rm -r $SS_SRC_DIR/obj
			;;
		"sm")
			rm -r $SM_SRC_DIR/obj
			;;
		"ii")
			rm -r $II_SRC_DIR/obj
			;;
		"mm")
			rm -r $MM_SRC_DIR/obj
			;;
		"pvc")
			rm -r $PVC_SRC_DIR/obj
			;;
		"pvr")
			rm -r $PVR_SRC_DIR/obj
			;;
		"wc")  
			rm -r $WC_SRC_DIR/obj
			;;
		"km")  
			rm -r $KM_SRC_DIR/obj
			;;
		"pr")  
			rm -r $PR1_SRC_DIR/obj
			rm -r $PR2_SRC_DIR/obj
			;;
		"pr1")  
			rm -r $PR1_SRC_DIR/obj
			;;
		"pr2")  
			rm -r $PR2_SRC_DIR/obj
			;;
		"rwr1")  
			rm -r $RWR1_SRC_DIR/obj
			;;
		"rwr2")  
			rm -r $RWR2_SRC_DIR/obj
			;;
		"l1n")  
			rm -r $L1N_SRC_DIR/obj
			;;
		"l2n")  
			rm -r $L2N_SRC_DIR/obj
			;;
		"sq")  
			rm -r $SQ_SRC_DIR/obj
			;;
		"sq2")  
			rm -r $SQ2_SRC_DIR/obj
			;;
		"smult")  
			rm -r $SMULT_SRC_DIR/obj
			;;
		"saxpy")  
			rm -r $SAXPY_SRC_DIR/obj
			;;
		"mv")  
			rm -r $MV_SRC_DIR/obj
			;;
		"mvw")  
			rm -r $MVW_SRC_DIR/obj
			;;
		"lz")  
			rm -r $LZ_SRC_DIR/obj
			;;
		"lzs")  
			rm -r $LZS_SRC_DIR/obj
			;;
		"cc1")  
			rm -r $CC1_SRC_DIR/obj
			;;
		"cc2")  
			rm -r $CC2_SRC_DIR/obj
			;;
		"cc3")  
			rm -r $CC3_SRC_DIR/obj
			;;
		"am")  
			rm -r $AM_SRC_DIR/obj
			;;
		"all")
			rm -r $SS_SRC_DIR/obj
			rm -r $SM_SRC_DIR/obj
			rm -r $II_SRC_DIR/obj
			rm -r $MM_SRC_DIR/obj
			rm -r $PVC_SRC_DIR/obj
			rm -r $PVR_SRC_DIR/obj
			rm -r $WC_SRC_DIR/obj
			rm -r $KM_SRC_DIR/obj
			;;
		*)
			echo $USAGE
			;;
	esac

#wrong arguments
else
	echo $USAGE
fi
