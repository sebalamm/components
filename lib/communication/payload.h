/******************************************************************************
 * payload.h
 *
 * Data structure for maintaining the (undirected) graph
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

#ifndef _PAYLOAD_H_
#define _PAYLOAD_H_

#include "config.h"

struct VertexPayload {
  VertexID deviate_;
  VertexID label_;
#ifdef TIEBREAK_DEGREE
  VertexID degree_;
#endif
  PEID root_;

  VertexPayload()
      : deviate_(MaxDeviate),
        label_(0),
#ifdef TIEBREAK_DEGREE
        degree_(0),
#endif
        root_(0) {}

  VertexPayload(VertexID deviate,
                VertexID label,
#ifdef TIEBREAK_DEGREE
                VertexID degree,
#endif
                PEID root)
      : deviate_(deviate),
        label_(label),
#ifdef TIEBREAK_DEGREE
        degree_(degree),
#endif
        root_(root) {}

  bool operator==(const VertexPayload &rhs) const {
    return std::tie(deviate_, 
                    label_, 
#ifdef TIEBREAK_DEGREE
                    rhs.degree_,
#endif
                    root_)
        == std::tie(rhs.deviate_, 
                    rhs.label_, 
#ifdef TIEBREAK_DEGREE
                    rhs.degree_,
#endif
                    rhs.root_);
  }

  bool operator!=(const VertexPayload &rhs) const {
    return !(*this == rhs);
  }
};

#endif
