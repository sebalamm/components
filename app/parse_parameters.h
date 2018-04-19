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
  conf.input_file = args.Get<std::string>("in", "in");
  conf.output_file = args.Get<std::string>("out", "out");
  conf.debug_output_file = args.Get<std::string>("debug_out", "tmp");

  // Benchmarks
  conf.iterations = args.Get<unsigned int>("i", 10);

  // Label propagation
  conf.prop_iterations = args.Get<unsigned int>("pi", 3);

  // Decomposition
  conf.beta = args.Get<double>("beta", 0.2);

  // Sequential computation
  conf.sequential_limit = args.Get<unsigned int>("seq", 1000);
}

#endif
