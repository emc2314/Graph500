#gengraph

[Koichi Shirahata](http://matsu-www.is.titech.ac.jp/~koichi-s/>) modified graph500(http://www.graph500.org/) reference implementation(Version 2.1.4, http://www.graph500.org/referencecode) for generating kronecker graphs into files, for measuring or debugging graph application with various graph sizes. Currently, graph500.c has been mogified for outputting an edge list of a kronecker graph into a text/binary file.

##Installation and Setup

    $ cd graph500-2.1.4  
    $ make

##Run the generator

To generate an edge list, just run one of below executables.

    -[seq|omp|...]-csr/{executable} [OPTION]

options:
    
    -o: output file name 
    -i: output binary file (in order of: src0 dst0 src1 dst1 ...)
        if not specified, output text file (default)

example for *text* edge list:

    ./seq-csr/sec-csr -s 14 -o s14.edge

example for *binary* edge list:

    ./seq-csr/sec-csr -s 14 -o s14.edge -i

**NOTE: if you do not use "-o" option, do not output and exit.**

Now you can use the edge list (e.g. s14.edge) for your graph applications.

##Open Source License
All Koichi Shirahata offered code is licensed under the [Boost Software License, Version 1.0](http://www.boost.org/LICENSE_1_0.txt) (See accompanying file `LICENSE_1_0.txt` or copy at `http://www.boost.org/LICENSE_1_0.txt`). And others follow the original license announcement (See COPYING)

##Copyright
* Copyright (C) 2013 [Koichi Shirahata](http://matsu-www.is.titech.ac.jp/~koichi-s/>) All Rights Reserved.

gengraph Webpage is https://github.com/koichi626/gengraph 