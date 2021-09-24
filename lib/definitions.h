/******************************************************************************
 * definitions.h
 *
 * Definition of basic types
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

#ifndef _GRAPH_DEFINITIONS_H_
#define _GRAPH_DEFINITIONS_H_

#include <vector> 
#include <limits> 

#define MPI_VERTEX MPI_UNSIGNED_LONG_LONG
// #define MPI_VERTEX MPI_UNSIGNED_LONG

// Constants
using PEID = int;
const PEID ROOT = 0;

// High/low prec
// typedef float HPFloat;
// typedef float LPFloat;
// typedef int LONG;
// typedef unsigned int ULONG;

using HPFloat = long double;
using LPFloat = double;
using LONG = long long;
using ULONG = unsigned long long;

// Graph access
using VertexID = ULONG;
using EdgeID = ULONG;

using VertexBuffer = std::vector<VertexID>;

// Graph constants
const int EmptyKey = -1;
const int DeleteKey = -2;
const VertexID MaxDeviate = std::numeric_limits<VertexID>::max() - 1;

// Message tags
const int CAGTag = 10;
const int ContractionTag = 100;
const int ExpTag = 1000;
const int CommTag = 10000;

#endif
