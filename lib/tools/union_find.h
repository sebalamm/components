/******************************************************************************
 * components.h *
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

#ifndef _UNION_FIND_H_
#define _UNION_FIND_H_

#include <vector>

#include "definitions.h"

class UnionFind {
 public:
  UnionFind(VertexID number_of_vertices) 
      : parent_(number_of_vertices), 
        rank_(number_of_vertices), 
        number_of_elements_(number_of_vertices) {
    for (VertexID v = 0; v < parent_.size(); ++v) {
      parent_[v] = v;
      rank_[v] = 0;
    }
  }

  inline void Union(VertexID lhs, VertexID rhs) {
    int set_lhs = Find(lhs);
    int set_rhs = Find(rhs);
    if (set_lhs != set_rhs) {
      if (rank_[set_lhs] < rank_[set_rhs]) {
        parent_[set_lhs] = set_rhs;
      } else {
        parent_[set_rhs] = set_lhs;
        if (rank_[set_lhs] == rank_[set_rhs]) rank_[set_lhs]++;
      }
      --number_of_elements_;
    }
  }

  inline VertexID Find(VertexID element) {
    if (parent_[element] != element) {
      VertexID set_element = Find(parent_[element]);  
      parent_[element] = set_element; // path compression
      return set_element;
    }
    return element;
  }

  inline VertexID GetNumberOfSets() const { return number_of_elements_; }

 private:
  std::vector<VertexID> parent_;
  std::vector<VertexID> rank_;

  // Number of elements in UF data structure.
  VertexID number_of_elements_;
};

#endif 
