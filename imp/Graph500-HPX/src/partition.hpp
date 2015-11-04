//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef GRAPH500_PARTITION_HPP
#define GRAPH500_PARTITION_HPP

#include <options.hpp>
#include <generator/make_graph.hpp>

#include <hpx/include/components.hpp>
#include <hpx/lcos/broadcast.hpp>

namespace graph500
{
    struct partition
      : hpx::components::managed_component_base<partition>
    {
        typedef std::vector<graph500::generator::packed_edge> edgelist_type;

        partition() {HPX_ASSERT(false);}
        partition(options const & opt)
          : opt_(opt)
        {
        }

        void init(std::size_t size, std::size_t rank)
        {
            std::cout << "size: " << size_ << " rank: " << rank_ << "\n";
            //edgelist_ = generator::make_graph(opt_.scale, desired_nedge, prng.userseed, prng.userseed, rank, size);
        }

        HPX_DEFINE_COMPONENT_ACTION(partition, init);

        options opt_;
        edgelist_type edgelist_;
        std::size_t size_;
        std::size_t rank_;
    };
}

HPX_REGISTER_ACTION_DECLARATION(
    graph500::partition::init_action
  , graph500_partition_init_action
)
HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_DECLARATION_2(
    graph500::partition::init_action
  , graph500_partition_init_action
)

#endif
