#include <ros/ros.h>
#include <rep_map/rep_map_base.hpp>

int main(int argc, char**argv){
	ros::init(argc, argv, "rep_map_base");

	ros::NodeHandle nh("~");

	std::string rep_map_pc_topic;
	nh.param<std::string>("rep_map_pc_topic", rep_map_pc_topic, "/rep_map/pc_map");

	std::string ref_frame;
	nh.param<std::string>("ref_frame", ref_frame, "camera_link");

	std::string fixed_frame;
	nh.param<std::string>("fixed_frame", fixed_frame, "base_link");

	std::string memory_frame("memory_frame");

	rep_map::RepMapBase amb(nh,
										rep_map_pc_topic,
										ref_frame,
										fixed_frame,
										memory_frame,
										1.0);

	ros::Rate rate(10);

	while(ros::ok()){
		amb.updateMap();
		amb.updateGridResults();
		amb.performQuery();
		amb.publishMap();
		amb.publishSpecificMap();
		ros::spinOnce();
		rate.sleep();
	}
}