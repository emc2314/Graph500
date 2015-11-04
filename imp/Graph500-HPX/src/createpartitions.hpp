//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef GRAPH500_CREATE_PARTITIONS_HPP
#define GRAPH500_CREATE_PARTITIONS_HPP

#include <hpx/config.hpp>
#include <options.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/locality_result.hpp>
#include <hpx/runtime/actions/plain_action.hpp>

#include <boost/range/algorithm/copy.hpp>

#include <utility>
#include <vector>

namespace graph500
{
    namespace detail
    {
        typedef
            std::pair<std::size_t, std::vector<hpx::util::remote_locality_result>>
            create_partitions_type;

        create_partitions_type
        create_partitions(
            std::vector<hpx::id_type> localities
          , hpx::components::component_type type
          , graph500::options const & opt);

        HPX_DEFINE_PLAIN_ACTION(create_partitions, create_partitions_action);

        inline std::size_t get_num_partitions()
        {
            return
                boost::lexical_cast<std::size_t>(
                    hpx::get_config_entry("graph500.num_partitions", "1")
                );
        }
    }

    std::vector<hpx::id_type> create_partitions(options const & opt);
}


HPX_REGISTER_PLAIN_ACTION_DECLARATION(graph500::detail::create_partitions_action)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    graph500::detail::create_partitions_type
  , graph500_create_partitions_type
);

#endif
