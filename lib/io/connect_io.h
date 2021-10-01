/*
 * Copyright 2016 Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file    graphReader.hpp
 * @ingroup group
 * @author  Chirag Jain <cjain7@gatech.edu>
 * @brief   Parallel graph reader using BLISS parallel IO support
 *
 * Copyright (c) 2015 Georgia Institute of Technology. All Rights Reserved.
 */

#ifndef PAR_GRAPH_READER_HPP 
#define PAR_GRAPH_READER_HPP

//Includes
#include <iostream>

//External includes
#include "io/file_loader.hpp"
#include "common/base_types.hpp"
#include "mxx/comm.hpp"


namespace conn 
{
  namespace graphGen
  {

    /*
     * @class     conn::graphGen::graphFileLoader
     * @brief     Enables parallel reading of the edgelist file
     */
    template<typename Iterator>
      class GraphFileLoader : public bliss::io::BaseFileParser<Iterator >
    {
      public:

        using RangeType = bliss::partition::range<size_t>;

        //Adjust the file loader's range by informing it how 
        //to find the beginning of the first record
        virtual std::size_t find_first_record(const Iterator &_data, const RangeType &parentRange, const RangeType &inMemRange, const RangeType &searchRange)
        {
          Iterator curr(_data);
          Iterator end(_data);

          RangeType r = RangeType::intersect(inMemRange, searchRange);

          std::size_t i = r.start;

          //curr is started at inMemRange by bliss, we need to move it to r.start
          std::advance(curr, i - inMemRange.start);

          //advance end by total size of inMemRange
          std::advance(end, inMemRange.size());

          //every rank except 0 skips initial partial or full record
          if(searchRange.start != parentRange.start)
          {
            this->findEOL(curr, end, i);
            this->findNonEOL(curr, end, i);
          }
          else
          {
            //skip initial sentences beginning with '%'
            //There is an assumption here that comments 
            //are very few and will fall in rank 0's partition

            //Jump to next line if we see a comment
            while(*curr == '%')
            {
              this->findEOL(curr, end, i);
              this->findNonEOL(curr, end, i);
            }
          }

          return i;
        }
    };


    /**
     * @class     conn::graphGen::GraphFileParser
     * @brief     Enables parallel reading of the edgelist file
     */
    template<typename Iterator, typename E>
      class GraphFileParser : public bliss::io::BaseFileParser<Iterator > 
    {
      private:

        //Type of base class
        using baseType = typename bliss::io::BaseFileParser<Iterator >;

        //MPI communicator
        mxx::comm comm;

        // Range stuff
        E &first_local_vertex;
        E &last_local_vertex;
        std::pair<E, E> &first_local_vertex_range;

        //Reference to the distributed edge list 
        std::vector< std::pair<E,E> > &edgeList;

        const static int OVERLAP = 50;

        std::string &filename;

      public:

        /**
         * @brief                 constructor for this class
         * @param[in] edgeList    Edgelist to build
         * @param[in] comm        mpi communicator
         */
        template <typename vID>
          GraphFileParser(std::vector< std::pair<vID, vID> > &edgeList, 
              vID &first_vertex, vID &last_vertex, std::pair<vID, vID> &first_vertex_range,
              std::string &filename, const mxx::comm &comm) 
          : 
          edgeList(edgeList), 
          first_local_vertex(first_vertex),
          last_local_vertex(last_vertex),
          first_local_vertex_range(first_vertex_range),
          filename(filename),
          comm(comm.copy())
        {
          static_assert(std::is_same<E, vID>::value, "Edge vector type should match");
        }

        /**
         * @brief                   populates the edge list vector 
         * @param[in]   filename    name of the file to read
         */
        void populateEdgeList()
        {
          //Value type over which Iterator is defined
          typedef typename std::iterator_traits<Iterator>::value_type IteratorValueType;

          //Define file loader type
          typedef bliss::io::FileLoader<IteratorValueType, OVERLAP, GraphFileLoader > FileLoaderType;

          FileLoaderType loader(filename, comm);

          //====  now process the file, one L1 block partition per MPI Rank 
          typename FileLoaderType::L1BlockType partition = loader.getNextL1Block();

          //To print the global file range
          //if (!comm.rank()) std::cout << "Global range " << loader.getFileRange() << std::endl;

          //Range of the file partition this MPI rank reads
          auto localFileRange = partition.getRange();

          //To print the local file range each rank reads
          //std::cout << "Rank " << comm.rank() << " , local file range" << localFileRange << std::endl;

          //Iterator over data in the file
          typename FileLoaderType::L1BlockType::iterator dataIter = partition.begin();

          //Initialize the byte offset counter over the range of this rank
          std::size_t i = localFileRange.start;

          bool lastEdgeRead = true;

          //Begin parsing the contents 
          //lastEdgeRead will be false when we reach the partition end boundary
          while (lastEdgeRead) {

            lastEdgeRead = readAnEdge(dataIter, partition.end(), i, localFileRange.end);
          }
        }

        /**
         * @brief             reads an edge assuming iterator points to 
         *                    beginning of a valid record
         * @param[in] curr    current iterator position
         * @return            true if edge is read successfully,
         *                    false if partition boundary is encountered 
         */
        template <typename Iter>
          bool readAnEdge(Iter& curr, const Iter &end, std::size_t& offset, std::size_t offsetEndRange) 
          {
            //make sure we point to non EOL value
            this->findNonEOL(curr, end, offset);

            //string to save the contents of a line
            std::string readLine;

            if(offset >= offsetEndRange || curr == end)
              return false;

            //till we see end of line
            while( ! (*curr == baseType::eol || *curr == baseType::cr)) 
            {
              //return if we are crossing the boundary
              if(curr == end)
                return false;

              //keep pushing the character to string
              readLine.append(curr, 1);

              //advance iterator
              curr++; offset++;

              //return if we are crossing the boundary
              if(curr == end)
                return false;
            }

            parseStringForEdge(readLine);

            return true;
          }

        /**
         * @brief             assumes string with two integers as input,
         *                    parse the integers and insert to edgeList
         * @param[in] record  string with 2 integers separated by space
         */                    
        inline void parseStringForEdge(std::string &record)
        {
          std::stringstream stream(record);

          size_t n = std::count(record.begin(), record.end(), ' ');

          char p;
          E vertex1, vertex2;

          stream >> p;
          stream >> vertex1;
          stream >> vertex2;

          if(p == 'e')
          {
            vertex1--;
            vertex2--;

            if (first_local_vertex == std::numeric_limits<VertexID>::max()) {
              first_local_vertex = vertex1;
            }
            last_local_vertex = vertex1;

            // Add edge(s)
            edgeList.emplace_back(vertex1, vertex2);
            edgeList.emplace_back(vertex2, vertex1);

            if (vertex1 == first_local_vertex) {
              if (first_local_vertex_range.first == std::numeric_limits<VertexID>::max()) {
                first_local_vertex_range.first = vertex2;
              } 
              first_local_vertex_range.second = vertex2;
            }


          }
        }

    };
  }
}

#endif
