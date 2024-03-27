/*********************************************************************
 *
 * Software License Agreement
 *
 *  Copyright (c) 2018, Simbe Robotics, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Simbe Robotics, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Steve Macenski (steven.macenski@simberobotics.com)
 * Purpose: convert native vdb files to pointclouds
 *********************************************************************/

// PCL
#include <pcl_ros/transforms.h>
// OpenVDB
#include <openvdb/openvdb.h>
// STL
#include <iostream>

namespace utilities
{

template<class T>
class VDB2PCLPointCloud
{
public:
  VDB2PCLPointCloud(){
    static_assert(std::is_base_of_v<openvdb::GridBase, T>);
    static_assert(!std::is_same_v<openvdb::GridBase, T>);
    openvdb::initialize();
  }
  
  void SetFile(const std::string& file_name){
    _file_name = file_name;
  }
  
  bool GetCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud){
    openvdb::io::File file(_file_name);
    file.open();
    openvdb::GridBase::Ptr baseGrid;
    openvdb::DoubleGrid::Ptr grid;

    bool valid_grid = false;

    for (openvdb::io::File::NameIterator nameIter = file.beginName();
                                  nameIter != file.endName(); ++nameIter)
    {
      if (nameIter.gridName() == "SpatioTemporalVoxelLayer")
      {
        baseGrid = file.readGrid(nameIter.gridName());
        grid = openvdb::gridPtrCast<openvdb::DoubleGrid>(baseGrid);
        valid_grid = true;
      }
    }

    if (!valid_grid)
    {
      std::cout << "No valid grid inside of provided file." << std::endl;
      return false;
    }

    //populate pcl pointcloud
    openvdb::DoubleGrid::ValueOnCIter cit_grid = grid->cbeginValueOn();
    for (cit_grid; cit_grid; ++cit_grid)
    {
      const openvdb::Vec3d pt = grid->indexToWorld(cit_grid.getCoord());
      cloud->push_back(pcl::PointXYZ(pt[0], pt[1], pt[2]));
    }

    return true;
  }

  void GetCloud(T& grid, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud){
    using CurrentValueOnCIter = typename T::ValueOnCIter;
    for (CurrentValueOnCIter iter = grid.cbeginValueOn(); iter; ++iter) {
      const openvdb::Vec3d pt = grid.indexToWorld(iter.getCoord());
      cloud->push_back(pcl::PointXYZ(pt[0], pt[1], pt[2]));
    }
  }

  void GetCloud(T& grid, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud){
    using CurrentValueOnCIter = typename T::ValueOnCIter;
    for (CurrentValueOnCIter iter = grid.cbeginValueOn(); iter; ++iter) {
      const openvdb::Vec3d pt = grid.indexToWorld(iter.getCoord());
      pcl::PointXYZI point;
      point.x = pt[0];
      point.y = pt[1];
      point.z = pt[2];
      point.intensity = iter.getValue();
      cloud->push_back(point);
    }
  }

private:
  std::string _file_name;
};

} // end namespace
