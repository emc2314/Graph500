//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef GRAPH500_OPTIONS_HPP
#define GRAPH500_OPTIONS_HPP

#include <hpx/config.hpp>

#include <cstdlib>
#include <string>

#include <boost/program_options.hpp>
#include <boost/preprocessor/stringize.hpp>

namespace graph500
{
    struct options
    {
        options()
          : verbose(false)
          , dumpname("")
          , rootname("")
          , nbfs(0)
          , scale(0)
          , edgefactor(0)
        {}

        options(boost::program_options::variables_map & vm)
          : verbose(vm["verbose"].as<bool>())
          , dump(vm["dump"].as<bool>())
          , dumpname(vm["dumpname"].as<std::string>())
          , rootname(vm["rootname"].as<std::string>())
          , nbfs(64)
          , scale(vm["scale"].as<std::int64_t>())
          , edgefactor(vm["edgefactor"].as<std::int64_t>())
        {}

        bool verbose;
        bool dump;
        std::string dumpname;
        std::string rootname;

        std::size_t nbfs;

        std::int64_t scale;
        std::int64_t edgefactor;

        template <typename Archive>
        void serialize(Archive & ar, unsigned /*version*/)
        {
            ar & verbose;
            ar & dump;
            ar & dumpname;
            ar & rootname;
            ar & nbfs;
            ar & scale;
            ar & edgefactor;
        }
    };

#define GRAPH500_RMAT_A 0.57
#define GRAPH500_RMAT_B 0.19
#define GRAPH500_RMAT_C 0.19
#define GRAPH500_RMAT_D (1.0 - (GRAPH500_RMAT_A + GRAPH500_RMAT_B + GRAPH500_RMAT_C)

    inline boost::program_options::options_description get_options()
    {
        boost::program_options::options_description
            desc("Usage: " HPX_APPLICATION_STRING " [options]");

        desc.add_options()
            (
                "verbose"
              , boost::program_options::value<bool>()->default_value(false)
              , "Enable extra (verbose) output"
            )
            (
                "dump"
              , boost::program_options::value<bool>()->default_value(false)
              , "Dump the generated graph (always true for generator)"
            )
            (
                "dumpname"
              , boost::program_options::value<std::string>()->default_value(std::string(""))
              , "Read the edge list from (or dump to) the named file"
            )
            (
                "rootname"
              , boost::program_options::value<std::string>()->default_value(std::string(""))
              , "Read the BFS roots from (or dump to) the named file"
            )
            (
                "scale"
              , boost::program_options::value<std::int64_t>()->default_value(14)
              , "scale (default 14)"
            )
            (
                "edgefactor"
              , boost::program_options::value<std::int64_t>()->default_value(16)
              , "edge factor (default 16)"
            )
        ;

        return desc;
    }
}

#endif
