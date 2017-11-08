/******************************************************************************
 * components.h
 *
 * Distributed computation of connected components
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
#ifndef _EDGE_HASH_H_
#define _EDGE_HASH_H_

#include <unordered_map>

#include "definitions.h"
#include <climits>

struct HashedEdge {
  VertexID k;
  VertexID source;
  VertexID target;
  PEID rank;
};

struct HashFunction {
  EdgeID operator()(const HashedEdge e) const {
    if (e.source < e.target) return e.source * e.k + e.target;
    else return e.target * e.k + e.source;
  }
};

struct EdgeComparator {
  bool operator()(const HashedEdge e1, const HashedEdge e2) const {
    bool eq = (e1.source == e2.source && e1.target == e2.target);
    return (eq || (e1.source == e2.target && e1.target == e2.source));
  }
};

typedef std::unordered_set<HashedEdge, HashFunction, EdgeComparator> EdgeHash;

#endif 
