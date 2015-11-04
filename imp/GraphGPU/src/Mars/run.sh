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

# GraphGPU bins
PR_BIN_DIR=$SDK_BIN_PATH/PageRank
RWR_BIN_DIR=$SDK_BIN_PATH/RWR
L1N_BIN_DIR=$SDK_BIN_PATH/L1norm
SMULT_BIN_DIR=$SDK_BIN_PATH/ScalarMult
SAXPY_BIN_DIR=$SDK_BIN_PATH/Saxpy
CC_BIN_DIR=$SDK_BIN_PATH/ConCmpt
CC3_BIN_DIR=$SDK_BIN_PATH/ConCmpt3
# Mars original bins
SS_BIN_DIR=$SDK_BIN_PATH/SimilarityScore
SM_BIN_DIR=$SDK_BIN_PATH/StringMatch
II_BIN_DIR=$SDK_BIN_PATH/InvertedIndex
PVC_BIN_DIR=$SDK_BIN_PATH/PageViewCount
PVR_BIN_DIR=$SDK_BIN_PATH/PageViewRank
MM_BIN_DIR=$SDK_BIN_PATH/MatrixMul
WC_BIN_DIR=$SDK_BIN_PATH/WordCount
KM_BIN_DIR=$SDK_BIN_PATH/Kmeans

# GraphGPU srcs
PR_SRC_DIR=$SDK_SRC_PATH/PageRank
RWR_SRC_DIR=$SDK_SRC_PATH/RWR
L1N_SRC_DIR=$SDK_SRC_PATH/L1norm
SMULT_SRC_DIR=$SDK_SRC_PATH/ScalarMult
SAXPY_SRC_DIR=$SDK_SRC_PATH/Saxpy
CC_SRC_DIR=$SDK_SRC_PATH/ConCmpt
CC3_SRC_DIR=$SDK_SRC_PATH/ConCmpt3
# Mars original srcs
SS_SRC_DIR=$SDK_SRC_PATH/SimilarityScore
SM_SRC_DIR=$SDK_SRC_PATH/StringMatch
II_SRC_DIR=$SDK_SRC_PATH/InvertedIndex
PVC_SRC_DIR=$SDK_SRC_PATH/PageViewCount
PVR_SRC_DIR=$SDK_SRC_PATH/PageViewRank
MM_SRC_DIR=$SDK_SRC_PATH/MatrixMul
WC_SRC_DIR=$SDK_SRC_PATH/WordCount
KM_SRC_DIR=$SDK_SRC_PATH/Kmeans

USAGE="usage: run.sh [make|run] [all|pr|rwr|cc|ss|sm|mm|pvc|pvr|ii|wc|km]"
STR_MAKE="making source code..."
STR_RUN="running test suite..."
STR_MAKE_PR="making PageRank source code..."
STR_MAKE_RWR="making RWR source code..."
STR_MAKE_CC="making ConCmpt source code..."
STR_MAKE_CC3="making ConCmpt3 source code..."
STR_MAKE_L1N="making L1norm source code..."
STR_MAKE_L2N="making L2norm source code..."
STR_MAKE_SMULT="making ScalarMult source code..."
STR_MAKE_SAXPY="making Saxpy source code..."
STR_MAKE_SS="making Similarity Score source code..."
STR_MAKE_SM="making String Match source code..."
STR_MAKE_MM="making MatrixMul source code..."
STR_MAKE_PVC="making PageViewCount source code..."
STR_MAKE_PVR="making PageViewRank source code..."
STR_MAKE_ALL="making all source code..."
STR_MAKE_II="making InvertedIndex source code..."
STR_MAKE_WC="making WordCount source code..."
STR_MAKE_KM="making Kmeans source code..."

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
		"l1n")
		        if [ ! -d $L1N_BIN_DIR ] 
			    then
			    echo "$L1N_BIN_DIR doesn't exist, mkdir it..."
			    mkdir $L1N_BIN_DIR
			fi
			cp -r $BIN_TMPL_PATH/L1N_BIN/* $L1N_BIN_DIR/
			cd $L1N_BIN_DIR
			sh run.sh
			;;
		"smult")
		        if [ ! -d $SMULT_BIN_DIR ] 
			    then
			    echo "$SMULT_BIN_DIR doesn't exist, mkdir it..."
			    mkdir $SMULT_BIN_DIR
			fi
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
		"cc3")
		        if [ ! -d $CC3_BIN_DIR ] 
			    then
			    echo "$CC3_BIN_DIR doesn't exist, mkdir it..."
			    mkdir $CC3_BIN_DIR
			fi
			cp -r $BIN_TMPL_PATH/CC3_BIN/* $CC3_BIN_DIR/
			cd $CC3_BIN_DIR
			sh run.sh
			;;
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
	        "pr")
			echo $STR_MAKE_PR
			MakeSrc $PR_BIN_DIR $PR_SRC_DIR
			;;
	        "rwr")
			echo $STR_MAKE_RWR
			MakeSrc $RWR_BIN_DIR $RWR_SRC_DIR
			echo $STR_MAKE_L1N
			MakeSrc $L1N_BIN_DIR $L1N_SRC_DIR
			echo $STR_MAKE_SMULT
			MakeSrc $SMULT_BIN_DIR $SMULT_SRC_DIR
			echo $STR_MAKE_SAXPY
			MakeSrc $SAXPY_BIN_DIR $SAXPY_SRC_DIR
			;;
	        "cc")
			echo $STR_MAKE_CC
			MakeSrc $CC_BIN_DIR $CC_SRC_DIR
			echo $STR_MAKE_CC3
			MakeSrc $CC3_BIN_DIR $CC3_SRC_DIR
			;;
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
		"all")
			echo $STR_MAKE_ALL
			MakeSrc $PR_BIN_DIR $PR_SRC_DIR
			MakeSrc $RWR_BIN_DIR $RWR_SRC_DIR
			MakeSrc $L1N_BIN_DIR $L1N_SRC_DIR
			MakeSrc $SMULT_BIN_DIR $SMULT_SRC_DIR
			MakeSrc $SAXPY_BIN_DIR $SAXPY_SRC_DIR
			MakeSrc $CC_BIN_DIR $CC_SRC_DIR
			MakeSrc $CC3_BIN_DIR $CC3_SRC_DIR
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
		"pr")  
			rm -r $PR_SRC_DIR/obj
			;;
		"rwr")  
			rm -r $RWR_SRC_DIR/obj
			rm -r $L1N_SRC_DIR/obj
			rm -r $SMULT_SRC_DIR/obj
			rm -r $SAXPY_SRC_DIR/obj
			;;
		"cc")  
			rm -r $CC_SRC_DIR/obj
			rm -r $CC3_SRC_DIR/obj
			;;
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
		"all")
			rm -r $PR_SRC_DIR/obj
			rm -r $RWR_SRC_DIR/obj
			rm -r $L1N_SRC_DIR/obj
			rm -r $SMULT_SRC_DIR/obj
			rm -r $SAXPY_SRC_DIR/obj
			rm -r $CC_SRC_DIR/obj
			rm -r $CC3_SRC_DIR/obj
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
