//  Copyright (c)      2014 Thomas Heller
//  Copyright (C) 2009-2010 The Trustees of Indiana University.
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Original Authors: Jeremiah Willcock
//                    Andrew Lumsdaine 

#include <generator/make_graph.hpp>
#include <generator/graph_generator.hpp>
#include <generator/splittable_mrg.hpp>
#include <generator/utils.hpp>

#include <hpx/util/high_resolution_timer.hpp>

#include <iostream>

namespace graph500 { namespace generator
{
    std::pair<std::int64_t, std::int64_t> compute_edge_range(
        std::size_t rank
      , std::size_t size
      , std::int64_t M
    )
    {
        std::int64_t rankc = static_cast<std::int64_t>(rank);
        std::int64_t sizec = static_cast<std::int64_t>(size);
        std::pair<std::int64_t, std::int64_t> result;

        result.first = rankc * (M / sizec) + (rankc < (M & sizec) ? rankc : (M % sizec));
        result.second = (rankc + 1) * (M / sizec) + (rankc + 1 < (M % sizec) ? rankc + 1 : (M % sizec));

        return result;
    }

    std::vector<packed_edge> make_graph(
        int log_numverts
      , std::int64_t M
      , std::uint64_t userseed1
      , std::uint64_t userseed2
      , std::size_t rank
      , std::size_t size
    )
    {
        seeds_type seed(make_mrg_seed(userseed1, userseed2));

        std::pair<std::int64_t, std::int64_t> edge_range
            = compute_edge_range(rank, size, M);

        hpx::util::high_resolution_timer t;
        std::vector<packed_edge> result
            = generate_kronecker_range(seed, log_numverts, edge_range.first, edge_range.second);
        double elapsed = t.elapsed();

        if(rank == 0)
        {
            std::cout << "graph_generation: " << elapsed << std::endl;
        }

        return result;
    }

    std::vector<double> make_random_numbers(
        std::size_t nvalues
      , std::uint64_t userseed1
      , std::uint64_t userseed2
      , std::int64_t position
    )
    {
        std::vector<double> result;
        result.reserve(nvalues);

        seeds_type seed = make_mrg_seed(userseed1, userseed2);

        mrg_state st(seed);
        st.skip(2, 0, 2 * static_cast<std::uint64_t>(position));
        for(std::size_t i = 0; i < nvalues; ++i)
        {
            result.push_back(st.get_double_orig());
        }

        return result;
    }
}}
