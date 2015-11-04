//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <partition.hpp>

typedef hpx::components::managed_component<graph500::partition> graph500_component_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(graph500_component_type, graph500_partition);

HPX_REGISTER_ACTION(
    graph500::partition::init_action
  , graph500_partition_init_action
)
HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_2(
    graph500::partition::init_action
  , graph500_partition_init_action
)
