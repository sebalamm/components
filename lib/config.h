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

#ifndef _GENERATOR_CONFIG_H_
#define _GENERATOR_CONFIG_H_

#include <string>
#include "definitions.h"

// Configuration for the generator.
struct Config {
  Config() {}
  virtual ~Config() {}

  // Seed for the PRNG
  int seed;
  // Input filename
  std::string input_file;
  // Output filename
  std::string output_file;
  // Debug output
  std::string debug_output_file;
  // Benchmarks
  unsigned int iterations;
};

#endif
