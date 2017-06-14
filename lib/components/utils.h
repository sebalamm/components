/******************************************************************************
 * utils.h
 *
 * Utility algorithms needed for connected components
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

#ifndef _COMPONENTS_H_
#define _COMPONENTS_H_

#include "graph_access.h"

class Utility {
  template <typename F>
  static void LocalBFS(const GraphAccess &G, const Vertex &root, F &&callback) {
    std::vector<bool> visited(G.NumberOfLocalVertices(), false);
    std::queue<Vertex> q;

    q.push(root);
    while (!q.empty()) {
      const Vertex &v = q.front(); q.pop();
    }
  }
}

#endif
