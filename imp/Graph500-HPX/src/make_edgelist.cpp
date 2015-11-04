//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <options.hpp>
#include <prng.hpp>
#include <generator/make_graph.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <fstream>

typedef std::vector<graph500::generator::packed_edge> edgelist_type;

int hpx_main(boost::program_options::variables_map & vm)
{
    {
        graph500::options opts(vm);

        std::int64_t nvtx_scale = 1L<<opts.scale;

        graph500::prng_state prng;

        std::int64_t desired_nedge = nvtx_scale * opts.edgefactor;

        // Catch a few possible overflows
        HPX_ASSERT(desired_nedge >= nvtx_scale);
        HPX_ASSERT(desired_nedge >= opts.edgefactor);

        if(opts.verbose) std::cerr << "Generating edge list... ";

        edgelist_type IJ;
        IJ = graph500::generator::make_graph(opts.scale, desired_nedge, prng.userseed, prng.userseed, 0, 1);

        if(opts.verbose) std::cerr << " done" << std::endl;

        int fd = 1;
        if(!opts.dumpname.empty())
        {
            fd = ::open(opts.dumpname.c_str(), O_WRONLY|O_CREAT|O_TRUNC, 0666);
        }
        if(fd < 0)
        {
            std::cerr << "Cannot open output file: " << opts.dumpname << "\n";
            hpx::terminate();
        }

        const char * data = reinterpret_cast<const char *>(&IJ[0]);
        ::write(fd, data, IJ.size() * sizeof(edgelist_type::value_type));
        ::close(fd);

        fd = 1;
        if(!opts.rootname.empty())
        {
            fd = ::open(opts.rootname.c_str(), O_WRONLY|O_CREAT|O_TRUNC, 0666);
            if(fd < 0)
            {
                std::cerr << "Cannot open output file: " << opts.dumpname << "\n";
                hpx::terminate();
            }

            std::vector<int> has_adj(nvtx_scale, 0);

            for(std::size_t k = 0; k < IJ.size(); ++k)
            {
                const std::int64_t i = IJ[k].v0();
                const std::int64_t j = IJ[k].v1();
                if(i != j)
                {
                    has_adj[i] = has_adj[j] = 1;
                }
            }

            std::int64_t t = 0;
            std::vector<std::int64_t> bfs_roots;
            bfs_roots.reserve(opts.nbfs);
            while(bfs_roots.size() < opts.nbfs && t < nvtx_scale)
            {
                double R = prng.state.get_double_orig();
                if(!has_adj[t] || (nvtx_scale - t) * R > opts.nbfs - bfs_roots.size())
                {
                    ++t;
                }
                else
                {
                    bfs_roots.push_back(t++);
                }
            }
            if(t >= nvtx_scale && bfs_roots.size() < opts.nbfs)
            {
                if(bfs_roots.size() > 0)
                {
                    std::cerr
                        << "Cannot find "
                        << opts.nbfs
                        << " sample roots of non-self degree > 0. using "
                        << bfs_roots.size() << "\n";
                }
                else
                {
                    std::cerr << "Cannot find any sample roots of non-self degree > 0.\n";
                    hpx::terminate();
                }
            }

            const char * data = reinterpret_cast<const char *>(&bfs_roots[0]);
            ::write(fd, data, bfs_roots.size() * sizeof(bfs_roots[0]));
            ::close(fd);
        }
    }
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(graph500::get_options(), argc, argv);
}
