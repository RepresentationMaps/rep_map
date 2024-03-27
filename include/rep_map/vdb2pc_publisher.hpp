#include <ros/ros.h>

#include <openvdb/openvdb.h>
#include <rep_map/vdb2pc.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/PointCloud2.h>

namespace ros_vdb_utilities{
	template<class T>
	class VDB2PCPublisher{
		protected:
			typedef pcl::PointCloud<pcl::PointXYZI> AttentionPointCloud;
		private:
			ros::NodeHandle n_;
			ros::Publisher pc_publisher_;

			utilities::VDB2PCLPointCloud<T> vdb_converter_;

			pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_;
		public:
			VDB2PCPublisher(ros::NodeHandle& n, const std::string& topic, const std::string& ref_frame):
				n_(n),
				pointcloud_(new AttentionPointCloud),
				pc_publisher_(n.advertise<AttentionPointCloud>(topic, 1)){
					pointcloud_->header.frame_id = ref_frame;
				}
			void publish(T& grid){
				pointcloud_->clear();
				vdb_converter_.GetCloud(grid, pointcloud_);

				pointcloud_->height = 1;
				pointcloud_->width = pointcloud_->size();
				pcl_conversions::toPCL(ros::Time::now(), pointcloud_->header.stamp);
				
				pc_publisher_.publish(*pointcloud_);
			}
			inline void updateRefFrame(const std::string& ref_frame){pointcloud_->header.frame_id = ref_frame;}
	};
};