/******************************************************************************
 * parse_parameters.h
 *
 * Parse I/O parameters and build config
 ******************************************************************************
 * Copyright (C) 2017 Sebastian Lamm <lamm@kit.edu>
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifndef _PARSE_PARAMETERS_H_
#define _PARSE_PARAMETERS_H_

#include "config.h"
#include "tools/arg_parser.h"

#include "definitions.h"

void ParseParameters(int argn, char **argv,
                     Config &conf) {
  ArgParser args(argn, argv);

  // RNG
  conf.seed = args.Get<int>("seed", 1);


  // I/O
  conf.input_type = args.Get<std::string>("input", "null");
  conf.input_file = args.Get<std::string>("in", "null");
  conf.output_file = args.Get<std::string>("out", "out");
  conf.debug_output_file = args.Get<std::string>("debug_out", "tmp");
  conf.print_verbose = args.IsSet("verbose");

  // Benchmarks
  conf.iterations = args.Get<unsigned int>("i", 10);

  // Label propagation
  conf.prop_iterations = args.Get<unsigned int>("pi", 3);

  // Shortcut propagation
  conf.number_of_hitters = args.Get<unsigned int>("heavy-hitters", 0);

  // Decomposition
  conf.beta = args.Get<double>("beta", 0.1);

  // Contraction
  conf.use_contraction = args.IsSet("contraction");
  conf.direct_contraction = args.IsSet("direct");
  conf.single_level_contraction = args.IsSet("single");

  // Sequential computation
  conf.sequential_limit = args.Get<unsigned int>("seq", 1);
  conf.fixed_limit = args.IsSet("fixed-seq");

  // High-degree vertices
  conf.degree_limit = args.Get<unsigned int>("deg", 100);
  conf.degree_threshold = args.Get<unsigned int>("hd-thres", 16);
  conf.neighborhood_sampling_factor = args.Get<float>("hd-sf", 0.5);

  // Vertex replication
  conf.replicate_high_degree = args.IsSet("replicate");

  // BFS
  conf.use_bfs = args.IsSet("bfs");

  // Communication
  conf.use_regular = args.IsSet("regular");

  // Memory
  conf.overallocate = args.IsSet("overalloc");

  // Generator
  conf.gen = args.Get<std::string>("gen", "null");
  conf.gen_k = args.Get<PEID>("k", 0);
  bool exact_n = args.IsSet("exact_n") || conf.gen == "grid_2d";
  if (exact_n)
	  conf.gen_n = args.Get<unsigned int>("n", 18);
  else 
	  conf.gen_n = 1 << args.Get<unsigned int>("n", 18);
  bool exact_m = args.IsSet("exact_m") || conf.gen == "grid_2d";
  if (exact_m)
	  conf.gen_m = args.Get<unsigned int>("m", 18);
  else 
	  conf.gen_m = 1 << args.Get<unsigned int>("m", 18);
  conf.gen_r = args.Get<float>("r", 0.001);
  conf.gen_gamma = args.Get<float>("gamma", 2.2);
  conf.gen_d = args.Get<float>("d", 16.0);
  conf.gen_p = args.Get<float>("p", 1.0);
  conf.gen_periodic = args.IsSet("periodic");
}

#endif
