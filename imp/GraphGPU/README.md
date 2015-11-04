#GraphGPU

[Koichi Shirahata](http://matsu-www.is.titech.ac.jp/~koichi-s/>) implemented high performance MapReduce-based graph processing applications running on GPUs. 

**Here is our paper**: Koichi Shirahata, Hitoshi Sato, Toyotaro Suzumura, and Satoshi Matsuoka. "[A Scalable Implementation of a MapReduce-based Graph Processing Algorithm for Large-scale Heterogeneous Supercomputers](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?tp=&arnumber=6546103)" *In Proceedings of the 13th IEEE/ACM International Symposium on Cluster, Cloud and Grid Computing (CCGrid 2013)*,  Delft, Netherlands, May 2013. 

This software modifies and includes [Mars](http://www.cse.ust.hk/gpuqp/Mars.html), a MapReduce framework on GPUs, developped by Bingsheng He et al.

##1. What is GraphGPU

GraphGPU is a MapReduce-based graph processing framework on GPU. The current distribution is based on Mars (a MapReduce framework on GPU).


##2. What is Provided

The GraphGPU distribution includes following directories:

* *src*: The source code of Mars framework and graph applications. Currently GraphGPU provides following applications: PageRank, Random Walk with Restart, Connected Components

* *gengraph*: Kronecker graph generator which is based on graph500 benchmark (http://www.graph500.org/). Generated edge list can be used as input for GraphGPU

* *edgelist*: Sample edge lists which can be used as input data of GraphGPU.

*Note: Current version only supports 1 GPU execution. We plan to add multi-GPU execution feature.*

##3. Installation and Setup

Make sure you have installed CUDA. Currently GraphGPU supports CUDA 4.0.

###3.1 Installation

    $ cd ${GRAPHGPU_HOME}/src/Mars
    $ vi run.sh
    Set ${SDK_PATH} in run.sh according to your installed CUDA SDK path by modifying run.sh (e.g. ${HOME}/NVIDIA_GPU_Computing_SDK). 
Optionally you can set ${SDK_BIN_PATH} and ${SDK_SRC_PATH} (default: SDK_BIN_PATH="$SDK_PATH/C/bin/linux/release" and SDK_SRC_PATH="$SDK_PATH/C/sample_apps").

    $ cp -r sample_apps ${SDK_PATH}/C
    $ cp run.sh ${SDK_SRC_PATH}


###3.2 Setup and build apps (show PageRank app as an example)

    $ cd ${SDK_PATH}/C/sample_apps
    $ ./run.sh make pr

###3.3 Run apps (show PageRank app as an example)

####Copy input edge list (only once)
    $ cp ${GRAPHGPU_HOME}/edgelist/* ${SDK_BIN_PATH}/PageRank

####Run apps
    $ ./run.sh run pr
    $ ./run.sh run pr 1 10



##4. Contact Information

Please contact [Koichi Shirahata](http://matsu-www.is.titech.ac.jp/~koichi-s/>) for further information. Please let us know about bug reports and suggested improvements.


##Open Source License
All Koichi Shirahata offered code is licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). And others follow the original license announcement.

##Copyright
* Copyright (C) 2014 [Koichi Shirahata](http://matsu-www.is.titech.ac.jp/~koichi-s/>) All Rights Reserved.

