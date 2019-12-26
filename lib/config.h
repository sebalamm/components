/******************************************************************************
 * config.h
 *
 * Configuration class to store user parameters
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

#ifndef _COMPONENT_CONFIG_H_
#define _COMPONENT_CONFIG_H_

#include <string>
#include "definitions.h"

// Configuration for the generator.
struct Config {
  Config() = default;
  virtual ~Config() = default;

  // Seed for the PRNG
  int seed{};
  // Input type
  std::string input_type;
  // Input filename
  std::string input_file;
  // Output filename
  std::string output_file;
  // Debug output
  std::string debug_output_file;
  // Number of vertices/edges
  VertexID n{};
  EdgeID m{};
  // High degree vertices
  VertexID degree_limit{};
  // Vertex replication
  bool replicate_high_degree;
  // Sequential computations
  VertexID sequential_limit{};
  // Benchmarks
  unsigned int iterations{};
  // Label propagation
  unsigned int prop_iterations{};
  // Decomposition
  double beta{};
  // Contraction
  bool use_contraction;
  bool direct_contraction;
  bool single_level_contraction;
  // Number of heavy hitters for shortcutting
  VertexID number_of_hitters;
  // Threshold for performing high degree splitting
  VertexID degree_threshold;
  // Threshold for neihgborhood sampling
  float neighborhood_sampling_factor;
  // Generator
  std::string gen;
  VertexID gen_n;
  EdgeID gen_m;
  float gen_r;
  float gen_p;
  bool gen_periodic;
  PEID gen_k;
  float gen_gamma;
  float gen_d;
};

#endif
