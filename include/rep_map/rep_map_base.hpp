#include <ros/ros.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <openvdb/openvdb.h>

#include <rep_plugins/rep_plugin_base.hpp>
#include <rep_map/AddPlugin.h>
#include <rep_map/RemovePlugin.h>
#include <rep_map/Query.h>
#include <rep_map/vdb2pc_publisher.hpp>

#include <std_msgs/UInt8MultiArray.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

#include <pluginlib/class_loader.h>

#include <map>
#include <functional>
#include <memory>
#include <type_traits>
#include <boost/shared_ptr.hpp>

#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>

#include <cv_bridge/cv_bridge.h>

#include <typeinfo>

#include <opencv2/opencv.hpp>

#define M_PI 3.14159265358979323846

namespace rep_map{
	namespace query_viz{
		void generateImageFromWordVectors(const std::vector<std::vector<std::string>>& wordVectors, 
										  ros::Publisher& image_pub) {
		    // Determine the size of the image based on the text lines
		    // int lineHeight = 30; // Adjust this value for desired line spacing
		    int imageWidth = 800; // Width of the image
		    int imageHeight = 800; // Image height
		    int numLines = wordVectors.size();
		    // int imageHeight = numLines * lineHeight;
		    int lineHeight = std::floor(imageHeight/numLines);

		    // Create a white image
		    cv::Mat image(imageHeight, imageWidth, CV_8UC3, cv::Scalar(255, 255, 255));

		    // Write text onto the image
		    int y = 0; // Starting y-coordinate for text
		    for (const auto& wordVector : wordVectors) {
		        std::string lineText;
		        for (const auto& word : wordVector) {
		            lineText += word + " ";
		        }
		        cv::putText(image, lineText, cv::Point(30, y+(lineHeight/2)), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
		        y += lineHeight;
		    }

		    cv_bridge::CvImage out_msg;
			out_msg.encoding = sensor_msgs::image_encodings::BGR8;
			out_msg.image = image;

			auto image_msg = out_msg.toImageMsg();
			image_pub.publish(image_msg);
		}
	}

	template<class... Ts> 
	struct overloaded : Ts... { using Ts::operator()...; };
	
	template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
	
	template<class... Ts>
	struct overloaded_or_no_op : Ts...
	{
	    using Ts::operator()...;
	
	    template<class... Us>
	    void operator()(Us...) const {  }
	};
	
	template<class... Ts> overloaded_or_no_op(Ts...) -> overloaded_or_no_op<Ts...>;

	enum CompareOperator{
		GREATER_THAN,
		LESS_THAN,
		EQUAL_TO,
		GREATER_OR_EQUAL_TO,
		LESS_OR_EQUAL_TO
	};

	enum FilteringOperator{
		FILTERING,
		COPYING
	};

	
	enum TreeType{
		FLOAT_TREE,
		INT32_TREE,
		BOOL_TREE,
		UNDEFINED
	};

	enum LocalMapOperation{
		BOOL_MASK
	};

	TreeType getTreeType(std::variant<openvdb::FloatTree, openvdb::Int32Tree, openvdb::BoolTree> tree){
		if (std::holds_alternative<openvdb::FloatTree>(tree)){
			return TreeType::FLOAT_TREE;
		}else if(std::holds_alternative<openvdb::Int32Tree>(tree)){
			return TreeType::INT32_TREE;
		}else if(std::holds_alternative<openvdb::BoolTree>(tree)){
			return TreeType::BOOL_TREE;
		}else{
			return TreeType::UNDEFINED; // Most-likely never entered
		}
	}

	using QueryTuple = std::tuple<TreeType, CompareOperator>;

	std::map<std::string, LocalMapOperation> local_map_operations{{"isInFieldOfView", LocalMapOperation::BOOL_MASK}};

	using ThresholdType = std::variant<float, int32_t, bool>;
	std::map<LocalMapOperation, std::pair<CompareOperator, ThresholdType>> local_operation_implementation = {{LocalMapOperation::BOOL_MASK, {CompareOperator::EQUAL_TO, true}}};
	
	std::map<std::string, QueryTuple> predicate_map{{"hasSaliencyLowerThan", {TreeType::FLOAT_TREE, CompareOperator::LESS_THAN}},
										  			{"hasSaliencyHigherThan", {TreeType::FLOAT_TREE, CompareOperator::GREATER_THAN}},
												  	{"hasSaliencyEqualTo", {TreeType::FLOAT_TREE, CompareOperator::EQUAL_TO}},
												  	{"containsObject", {TreeType::INT32_TREE, CompareOperator::EQUAL_TO}}};

	std::map<std::vector<std::string>, openvdb::GridBase::Ptr> starting_point_map;

	std::map<std::vector<std::string>, TreeType> associated_map{{{"as", "saliency"}, TreeType::FLOAT_TREE},
																{{"as", "object_class"}, TreeType::INT32_TREE}};

	std::map<std::string, CompareOperator> predicate_operator_map{{"hasSaliencyLowerThan", CompareOperator::LESS_THAN},
																  {"hasSaliencyHigherThan", CompareOperator::GREATER_THAN},
																  {"hasSaliencyEqualTo", CompareOperator::EQUAL_TO},
																  {"containsObject", CompareOperator::EQUAL_TO}};

	std::map<std::string, std::vector<std::string>> predicate_second_term_map{{"hasSaliencyLowerThan", {"as", "saliency"}},
																			  {"hasSaliencyHigherThan", {"as", "saliency"}},
																			  {"hasSaliencyEqualTo", {"as", "saliency"}},
																			  {"containsObject", {"as", "object_class"}}};																  

	template<class T>
	static constexpr bool check_grid_type_validity(){
		constexpr bool check_base = std::is_base_of_v<openvdb::GridBase, T>;
		constexpr bool is_not_base = !std::is_same_v<openvdb::GridBase, T>;
		return check_base && is_not_base;
	}

	template<class T>
	static constexpr bool check_tree_type_validity(){
		constexpr bool check_base = std::is_base_of_v<openvdb::TreeBase, T>;
		constexpr bool is_not_base = !std::is_same_v<openvdb::TreeBase, T>;
		return check_base && is_not_base;
	}

	template<class T, class U, class V>
	static void filterMap2(T& tree_a,
				   	U& tree_b,
				   	T& tree_ret,
				   	const V& threshold_value,
				   	const CompareOperator filter_operator){
		if constexpr (check_tree_type_validity<T>() && check_tree_type_validity<U>()){
			using VoxelValueTypeA = typename T::ValueType;
			using VoxelValueTypeB = typename U::ValueType;

			if constexpr (std::is_same_v<VoxelValueTypeB, V>){
				struct Local {
					static inline void op(openvdb::CombineArgs<VoxelValueTypeA, VoxelValueTypeB>& args, const V& th_value, const CompareOperator& filter_op){
						args.setResult(args.a());
						bool set;
						switch (filter_op){
							case CompareOperator::GREATER_THAN:
								set = args.b() > th_value;
								break;
							case CompareOperator::LESS_THAN:
								set = args.b() < th_value;
								break;
							case CompareOperator::EQUAL_TO:
								set = args.b() == th_value;
								break;
							case CompareOperator::GREATER_OR_EQUAL_TO:
								set = args.b() >= th_value;
								break;
							case CompareOperator::LESS_OR_EQUAL_TO:
								set = args.b() <= th_value;
								break;
							default:
								set = false;
						}
						args.setResultIsActive(args.aIsActive() && args.bIsActive() && set);
					}
				};

				auto opWithAdditionalArg = [&](openvdb::CombineArgs<VoxelValueTypeA, VoxelValueTypeB>& args){
		        	// Call the original functor with additional argument
		        	Local::op(args, threshold_value, filter_operator);
		    	};

		    	tree_ret.combine2Extended(tree_a, tree_b, opWithAdditionalArg);
			}
		}
	}

	void removeCharacters(std::string& str, char target) {
	    for (int i = str.size() - 1; i >= 0; --i) {
	        if (str[i] == target) {
	            str.erase(i, 1);
	        } else {
	            break;
	        }
	    }
	}

	template<class T>
	bool checkThresholdFormat(std::string elem){
	    if constexpr (std::is_same_v<T, int>){
	        return (std::to_string(std::stoi(elem)) == elem);
	    }else if constexpr(std::is_same_v<T, float>){
	        auto reconstructed_elem = std::to_string(std::stof(elem));
	        removeCharacters(reconstructed_elem, '0');
	        return (reconstructed_elem == elem);
	    }
	}

	class RepMapBase{
		protected:

			class QueryFunction{
				using grid_t = std::variant<openvdb::FloatGrid::Ptr, openvdb::Int32Grid::Ptr, openvdb::BoolGrid::Ptr>;
				using leaf_t = std::variant<float, int32_t, bool>;

				grid_t tree_a_;
				grid_t tree_b_;
				grid_t tree_ret_;
				leaf_t threshold_;
				CompareOperator comp_op_;

				bool constant_threshold_;

			public:
				QueryFunction(){
					constant_threshold_ = false;
				}

				QueryFunction(grid_t tree_a, grid_t tree_b, grid_t tree_ret,
							  leaf_t threshold, CompareOperator comp_op, std::string predicate):
					tree_a_(tree_a), tree_b_(tree_b), tree_ret_(tree_ret),
					threshold_(threshold), comp_op_(comp_op){
						constant_threshold_ = false;
					}

				QueryFunction(CompareOperator comp_op):comp_op_(comp_op){
					constant_threshold_ = false;
				}

				void setGridA(grid_t grid_a){
					tree_a_ = grid_a;
				}

				void setGridB(grid_t grid_b){
					tree_b_ = grid_b;
				}

				void setGridRet(grid_t grid_ret){
					tree_ret_ = grid_ret;
				}

				void setThreshold(leaf_t threshold){
					threshold_ = threshold;
				}

				void setConstantThreshold(){
					constant_threshold_ = true;
				}

				bool const isConstantThreshold(){
					return constant_threshold_;
				}

				void performQuery(){
					auto visit_f = overloaded_or_no_op{[=](openvdb::FloatGrid::Ptr grid1, openvdb::FloatGrid::Ptr grid2, openvdb::FloatGrid::Ptr grid_out, float threshold) {
															std::cout<<"Executing! float float float"<<std::endl;
															filterMap2(grid1->tree(), grid2->tree(), grid_out->tree(), threshold, comp_op_); },
											  		   [=](openvdb::FloatGrid::Ptr grid1, openvdb::Int32Grid::Ptr grid2, openvdb::FloatGrid::Ptr grid_out, int threshold) {
											  		   		std::cout<<"Executing! float int float"<<std::endl;
											  		   		filterMap2(grid1->tree(), grid2->tree(), grid_out->tree(), threshold, comp_op_); },
											  		   [=](openvdb::Int32Grid::Ptr grid1, openvdb::Int32Grid::Ptr grid2, openvdb::Int32Grid::Ptr grid_out, int threshold) {
											  		   		std::cout<<"Executing! int int int"<<std::endl;
											  		   		filterMap2(grid1->tree(), grid2->tree(), grid_out->tree(), threshold, comp_op_); },
											  		   [=](openvdb::Int32Grid::Ptr grid1, openvdb::FloatGrid::Ptr grid2, openvdb::Int32Grid::Ptr grid_out, float threshold) {
											  		   		std::cout<<"Executing! int float int"<<std::endl;
											  		   		filterMap2(grid1->tree(), grid2->tree(), grid_out->tree(), threshold, comp_op_); },
											  		   [=](openvdb::Int32Grid::Ptr grid1, openvdb::BoolGrid::Ptr grid2, openvdb::Int32Grid::Ptr grid_out, bool threshold) {
											  		   		std::cout<<"Executing! int bool int"<<std::endl;
											  		   		filterMap2(grid1->tree(), grid2->tree(), grid_out->tree(), threshold, comp_op_); },
											  		   [=](openvdb::FloatGrid::Ptr grid1, openvdb::BoolGrid::Ptr grid2, openvdb::FloatGrid::Ptr grid_out, bool threshold) {
											  		   		std::cout<<"Executing! float bool float"<<std::endl;
											  		   		filterMap2(grid1->tree(), grid2->tree(), grid_out->tree(), threshold, comp_op_); }};

					visit(visit_f, tree_a_, tree_b_, tree_ret_, threshold_);
				}
			};

			std::map<std::string, QueryFunction> queries_map_;

			std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
			std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

			float fov_;
			std::string ref_frame_;
			std::string fixed_frame_;
			std::string memory_frame_; // Frame coherent with the robot's base frame
									   // in terms of translation and with the fixed
									   // frame in terms of rotation

			typedef boost::shared_ptr<rep_plugins::PluginBase> AttentionPluginPtr;

			bool loadPlugin(const std::string& plugin_name);
			bool unloadPlugin(std::string& plugin_name);

			void compScreen(openvdb::FloatTree& attention_tree, openvdb::FloatTree& local_tree);
			void compIntComb(openvdb::Int32Tree& attention_tree, openvdb::Int32Tree& local_tree);
			void compBooleanMask(openvdb::Int32Tree& tree_a,
								 openvdb::BoolTree& tree_b,
								 openvdb::Int32Tree& tree_ret);

			static std::vector<std::string> const splitString(const std::string& str, char delimiter) {
			    std::vector<std::string> tokens;
			    std::stringstream ss(str);
			    std::string token;
			    
			    while (std::getline(ss, token, delimiter)) {
			        tokens.push_back(token);
			    }
			    
			    return tokens;
			}

			static std::vector<std::vector<std::string>> const parseQuery(const std::string& query){
				std::vector<std::string> lines = splitString(query, ',');
				std::vector<std::vector<std::string>> decomposed_query;
				for (auto& line: lines){
					std::vector<std::string> words = splitString(line, ' ');
					if (words.size() == 3)
						decomposed_query.push_back(words);
				}
				return decomposed_query;
			}

			template<class T>
			static void filterObject(T& attention_tree,
							  openvdb::Int32Tree& local_tree,
							  const int32_t& object_id,
							  const FilteringOperator& filtering_operator = FilteringOperator::FILTERING){
				if constexpr (check_tree_type_validity<T>()){
					using VoxelValueType = typename T::ValueType;

					struct Local {
						static inline void op(openvdb::CombineArgs<VoxelValueType, int32_t>& args, const int32_t& object_id, const FilteringOperator& filtering_op){
							switch (filtering_op){
								case FilteringOperator::FILTERING:
									args.setResult(args.a());
									break;
								case FilteringOperator::COPYING:
									args.setResult(args.b());
									break;
							}
							args.setResultIsActive(args.bIsActive() && (args.b() == object_id));
						}
					};

					auto opWithAdditionalArg = [&](openvdb::CombineArgs<VoxelValueType, int32_t>& args){
			        	// Call the original functor with additional argument
			        	Local::op(args, object_id, filtering_operator);
			    	};

					attention_tree.combineExtended(local_tree, opWithAdditionalArg);
				}
			}

			template<class T>
			static void filterObject2(T& tree_a,
							   openvdb::Int32Tree& tree_b,
							   T& tree_ret,
							   const int32_t& object_id){
				if constexpr (check_tree_type_validity<T>()){
					using VoxelValueType = typename T::ValueType;

					struct Local {
						static inline void op(openvdb::CombineArgs<VoxelValueType, int32_t>& args, const int32_t& object_id){
							args.setResult(args.a());
							args.setResultIsActive(args.bIsActive() && (args.b() == object_id));
						}
					};

					auto opWithAdditionalArg = [&](openvdb::CombineArgs<VoxelValueType, int32_t>& args){
			        	// Call the original functor with additional argument
			        	Local::op(args, object_id);
			    	};

					tree_ret.combine2Extended(tree_a, tree_b, opWithAdditionalArg, /*prune=*/false);
				}
			}		

			template<class T>
			static void filterSaliency(T& attention_tree,
								openvdb::FloatTree& local_tree,
								const float& saliency,
								const CompareOperator& sal_operator,
								const FilteringOperator& filtering_operator = FilteringOperator::FILTERING){
				if constexpr (check_tree_type_validity<T>()){
					using VoxelValueType = typename T::ValueType;

					struct Local {
						static inline void op(openvdb::CombineArgs<VoxelValueType, float>& args, const float& saliency, const CompareOperator& sal_operator, const FilteringOperator& filtering_op){
							switch (filtering_op){
								case FilteringOperator::FILTERING:
									args.setResult(args.a());
									break;
								case FilteringOperator::COPYING:
									args.setResult(args.b());
									break;
							}
							bool set;
							switch (sal_operator){
								case CompareOperator::GREATER_THAN:
									set = args.b() > saliency;
								const FilteringOperator& filtering_operator = FilteringOperator::FILTERING;
									break;
								case CompareOperator::LESS_THAN:
									set = args.b() < saliency;
									break;
								case CompareOperator::EQUAL_TO:
									set = args.b() == saliency;
									break;
								case CompareOperator::GREATER_OR_EQUAL_TO:
									set = args.b() >= saliency;
									break;
								case CompareOperator::LESS_OR_EQUAL_TO:
									set = args.b() <= saliency;
									break;
								default:
									set = false;
							}
							args.setResultIsActive(args.bIsActive() && set);
						}
					};

					auto opWithAdditionalArg = [&](openvdb::CombineArgs<VoxelValueType, float>& args){
			        	// Call the original functor with additional argument
			        	Local::op(args, saliency, sal_operator, filtering_operator);
			    	};

					attention_tree.combineExtended(local_tree, opWithAdditionalArg);
				}
			}

			template<class T>
			static void filterSaliency2(T& tree_a,
								 openvdb::FloatTree& tree_b,
								 T& tree_ret,
								 const float& saliency,
								 const CompareOperator& sal_operator){
				if constexpr (check_tree_type_validity<T>()){
					using VoxelValueType = typename T::ValueType;

					struct Local {
						static inline void op(openvdb::CombineArgs<VoxelValueType, float>& args, const float& saliency, const CompareOperator& sal_operator){
							args.setResult(args.a());
							bool set;
							switch (sal_operator){
								case CompareOperator::GREATER_THAN:
									set = args.b() > saliency;
									break;
								case CompareOperator::LESS_THAN:
									set = args.b() < saliency;
									break;
								case CompareOperator::EQUAL_TO:
									set = args.b() == saliency;
									break;
								case CompareOperator::GREATER_OR_EQUAL_TO:
									set = args.b() >= saliency;
									break;
								case CompareOperator::LESS_OR_EQUAL_TO:
									set = args.b() <= saliency;
									break;
								default:
									set = false;
							}
							args.setResultIsActive(args.aIsActive() && args.bIsActive() && set);
						}
					};

					auto opWithAdditionalArg = [&](openvdb::CombineArgs<VoxelValueType, float>& args){
			        	// Call the original functor with additional argument
			        	Local::op(args, saliency, sal_operator);
			    	};

					tree_ret.combine2Extended(tree_a, tree_b, opWithAdditionalArg);
				}
			}

			template<class T, class U, class V>
			static void filterMap(T& tree_a,
						   U& tree_b,
						   const V& threshold_value,
						   const CompareOperator filter_operator){
				if constexpr (check_tree_type_validity<T>() && check_tree_type_validity<U>()){
					using VoxelValueTypeA = typename T::ValueType;
					using VoxelValueTypeB = typename U::ValueType;

					if constexpr (std::is_same_v<VoxelValueTypeB, V>){
						struct Local {
							static inline void op(openvdb::CombineArgs<VoxelValueTypeA, VoxelValueTypeB>& args, const V& th_value, const CompareOperator& filter_op){
								args.setResult(args.a());
								bool set;
								switch (filter_op){
									case CompareOperator::GREATER_THAN:
										set = args.b() > th_value;
										break;
									case CompareOperator::LESS_THAN:
										set = args.b() < th_value;
										break;
									case CompareOperator::EQUAL_TO:
										set = args.b() == th_value;
										break;
									case CompareOperator::GREATER_OR_EQUAL_TO:
										set = args.b() >= th_value;
										break;
									case CompareOperator::LESS_OR_EQUAL_TO:
										set = args.b() <= th_value;
										break;
									default:
										set = false;
								}
								args.setResultIsActive(args.aIsActive() && args.bIsActive() && set);
							}
						};

						auto opWithAdditionalArg = [&](openvdb::CombineArgs<VoxelValueTypeA, VoxelValueTypeB>& args){
				        	// Call the original functor with additional argument
				        	Local::op(args, threshold_value, filter_operator);
				    	};

						tree_a.combineExtended(tree_b, opWithAdditionalArg);
					}
				}
			}	

			std::vector<openvdb::Coord> findHighestValueCoord();
		private:
			std::map<std::string, AttentionPluginPtr> plugins_;
			std::map<std::string, openvdb::FloatGrid::Ptr> local_maps_;

			openvdb::FloatGrid::Ptr attention_map_;
			std::map<std::string, openvdb::GridBase::Ptr> full_maps_;
			std::vector<rep_plugins::GridResult> local_grid_results_;
			std::vector<rep_plugins::GridResult> processed_grid_results_;
			std::variant<openvdb::FloatGrid::Ptr, openvdb::Int32Grid::Ptr, openvdb::BoolGrid::Ptr> queried_map_;

			ros::NodeHandle n_;

			ros::ServiceServer add_plugin_service_;
			ros::ServiceServer remove_plugin_service_;
			ros::ServiceServer query_service_;

			pluginlib::ClassLoader<rep_plugins::PluginBase> plugin_loader_;

			ros_vdb_utilities::VDB2PCPublisher<openvdb::FloatGrid> pc_publisher_;
			ros_vdb_utilities::VDB2PCPublisher<openvdb::FloatGrid> pc_publisher_saliency_;
			ros_vdb_utilities::VDB2PCPublisher<openvdb::Int32Grid> pc_publisher_objects_;
			ros_vdb_utilities::VDB2PCPublisher<openvdb::FloatGrid> float_queried_map_publisher_;
			ros_vdb_utilities::VDB2PCPublisher<openvdb::Int32Grid> int_queried_map_publisher_;

			// Tools for querying
			bool result_;
			std::vector<std::vector<std::string>> processed_query_;
			ros::Publisher query_image_pub;

			// Scoring debugging functions
			void printCircle(sensor_msgs::ImagePtr image, std::vector<openvdb::Vec3d> top_k_world, const int& width, const int& height);

			bool addPlugin(rep_map::AddPlugin::Request& req,
						   rep_map::AddPlugin::Response& res);
			bool removePlugin(rep_map::RemovePlugin::Request& req,
						   	  rep_map::RemovePlugin::Response& res);
			bool processQuery(rep_map::Query::Request& req,
							 rep_map::Query::Response& res);
		public:
			RepMapBase(ros::NodeHandle& n,
							 const std::string& pc_topic="/attention_map/map_pc",
							 const std::string& ref_frame="map",
							 const std::string& fixed_frame="map",
							 const std::string& memory_frame="memory_frame",
							 const float& fov = M_PI*1.5);
			void updateMap();
			void updateGridResults();
			inline void publishMap(){pc_publisher_.publish(*attention_map_);}
			inline void publishSpecificMap(){
				std::vector<std::vector<std::string>> saliency_data_format = {{"hasSaliencyLowerThan"}, {"hasSaliencyHigherThan"}, {"as", "saliency"}};
				std::vector<std::vector<std::string>> object_data_format = {{"containsObject"}, {"as", "object_class"}};
				for (auto& grid: processed_grid_results_){
					if (grid.data_format_ == saliency_data_format){
						pc_publisher_saliency_.publish(*(openvdb::gridPtrCast<openvdb::FloatGrid>(grid.grid_ptr_)));
					}else if(grid.data_format_ == object_data_format){
						pc_publisher_objects_.publish(*(openvdb::gridPtrCast<openvdb::Int32Grid>(grid.grid_ptr_)));
					}
				}
				if (std::holds_alternative<openvdb::FloatGrid::Ptr>(queried_map_)){
					auto map_to_publish = std::get<openvdb::FloatGrid::Ptr>(queried_map_);
					if(map_to_publish)
						float_queried_map_publisher_.publish(*map_to_publish);
				}else if (std::holds_alternative<openvdb::Int32Grid::Ptr>(queried_map_)){
					auto map_to_publish = std::get<openvdb::Int32Grid::Ptr>(queried_map_);
					if(map_to_publish)
						int_queried_map_publisher_.publish(*map_to_publish);
				}
			}

			openvdb::GridBase::Ptr findMapByPredicate(const std::vector<std::string> predicate){
				for (auto grid: processed_grid_results_){
					for (auto& grid_predicate: grid.data_format_){
						if (grid_predicate == predicate){
							return grid.grid_ptr_;
						}
					}
				}
				return nullptr;
			}
			openvdb::GridBase::Ptr findLocalMapByPredicate(const std::vector<std::string> predicate){
				for (auto grid: local_grid_results_){
					for (auto& grid_predicate: grid.data_format_){
						if (grid_predicate == predicate){
							return grid.grid_ptr_;
						}
					}
				}
				return nullptr;
			}
			void performQuery();
	};
};