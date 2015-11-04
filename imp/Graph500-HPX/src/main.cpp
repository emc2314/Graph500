//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <createpartitions.hpp>
#include <options.hpp>
#include <partition.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

int hpx_main(boost::program_options::variables_map & vm)
{
    graph500::options opts(vm);

    std::vector<hpx::id_type> partitions = create_partitions(opts);

    hpx::lcos::broadcast_with_index<graph500::partition::init_action>(partitions, partitions.size()).wait();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(graph500::get_options(), argc, argv);
}
