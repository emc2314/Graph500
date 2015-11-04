//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <createpartitions.hpp>
#include <partition.hpp>

#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/lcos/wait_all.hpp>

#include <boost/serialization/vector.hpp>

typedef graph500::detail::create_partitions_action action_type;

HPX_REGISTER_PLAIN_ACTION(
    graph500::detail::create_partitions_action
  , graph500_create_partitions_action);
HPX_REGISTER_BASE_LCO_WITH_VALUE(
    graph500::detail::create_partitions_type
  , graph500_create_partitions_type
);

namespace graph500
{
    namespace detail
    {
        create_partitions_type
        create_partitions(
            std::vector<hpx::id_type> localities
          , hpx::components::component_type type
          , graph500::options const & opt)
        {
            typedef hpx::util::remote_locality_result value_type;
            
            create_partitions_type res;
            if(localities.size() == 0)
            {
                return res;
            }

            hpx::id_type this_loc = localities[0];

            typedef
                hpx::components::server::create_component_action1<partition, graph500::options>
                create_action_type;

            std::size_t num_components = get_num_partitions();

            typedef
                std::vector<hpx::unique_future<hpx::naming::gid_type>>
                futures_type;

            std::vector<hpx::unique_future<create_partitions_type>> components;
            components.reserve(2);

            if(localities.size() > 1)
            {
                std::size_t half = (localities.size() / 2) + 1;
                std::vector<hpx::id_type>
                    locs_first(localities.begin() + 1, localities.begin() + half);
                std::vector<hpx::id_type>
                    locs_second(localities.begin() + half, localities.end());

                if(!locs_first.empty())
                {
                    hpx::lcos::packaged_action<action_type, create_partitions_type> p;
                    hpx::id_type id = locs_first[0];
                    p.apply(hpx::launch::async, id, std::move(locs_first), type, opt);
                    components.push_back(p.get_future());
                }
                if(!locs_second.empty())
                {
                    hpx::lcos::packaged_action<action_type, create_partitions_type> p;
                    hpx::id_type id = locs_second[0];
                    p.apply(hpx::launch::async, id, std::move(locs_second), type, opt);
                    components.push_back(p.get_future());
                }
            }
            
            futures_type fs;
            fs.reserve(num_components);
            for(std::size_t i = 0; i < num_components; ++i)
            {
                typedef
                    hpx::lcos::packaged_action<
                        create_action_type
                      , hpx::naming::gid_type
                    >
                    packaged_action;
                packaged_action p;
                p.apply(hpx::launch::async, this_loc, opt);
                fs.push_back(p.get_future());
            }

            res.first = num_components;
            res.second.push_back(value_type(this_loc.get_gid(), type));
            res.second.back().gids_.clear();
            res.second.back().gids_.reserve(num_components);
            BOOST_FOREACH(hpx::unique_future<hpx::naming::gid_type> & f, fs)
            {
                res.second.back().gids_.push_back(f.get());
            }

            hpx::wait_all(components);

            BOOST_FOREACH(hpx::unique_future<create_partitions_type> & f, components)
            {
                create_partitions_type r = f.get();
                res.second.insert(res.second.end(), r.second.begin(), r.second.end());
                res.first += r.first;
            }
            return res;
        }
    }

    std::vector<hpx::id_type> create_partitions(options const & opt)
    {
        hpx::components::component_type type =
            hpx::components::get_component_type<partition>();

        std::vector<hpx::id_type> localities = hpx::find_all_localities(type);

        hpx::id_type id = localities[0];

        hpx::unique_future<detail::create_partitions_type> async_result
            = hpx::async(action_type(), id, std::move(localities), type, opt);

        std::vector<hpx::id_type> components;

        detail::create_partitions_type result = async_result.get();
        std::size_t num_components = result.first;
        components.reserve(num_components);

        std::vector<hpx::util::locality_result> res;
        res.reserve(num_components);
        boost::copy(result.second, std::back_inserter(res));
        boost::copy(hpx::util::locality_results(res), std::back_inserter(components));

        return components;
    }
}

