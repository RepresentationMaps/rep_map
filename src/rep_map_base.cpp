#include <rep_map/rep_map_base.hpp>

#include <openvdb/openvdb.h>
#include <openvdb/tools/Composite.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>

namespace rep_map{
	RepMapBase::RepMapBase(ros::NodeHandle& n,
						   const std::string& pc_topic,
						   const std::string& ref_frame,
						   const std::string& fixed_frame,
						   const std::string& memory_frame,
						   const float& fov):
	n_(n),
	plugin_loader_("rep_plugins", "rep_plugins::PluginBase"),
	add_plugin_service_(n.advertiseService("/rep_map/add_plugin", &RepMapBase::addPlugin, this)),
	remove_plugin_service_(n.advertiseService("/rep_map/remove_plugin", &RepMapBase::removePlugin, this)),
	query_service_(n.advertiseService("/rep_map/query", &RepMapBase::processQuery, this)),
	query_image_pub(n.advertise<sensor_msgs::Image>("/rep_map/query/viz", 1)),
	pc_publisher_(n, pc_topic, ref_frame),
	pc_publisher_saliency_(n, "/vwm/robot/saliency", ref_frame),
	pc_publisher_objects_(n, "/vwm/robot/objects", ref_frame),
	float_queried_map_publisher_(n, "/vwm/robot/queried", ref_frame),
	int_queried_map_publisher_(n, "/vwm/robot/queried", ref_frame),
	fov_(fov),
	ref_frame_(ref_frame),
	fixed_frame_(fixed_frame),
	memory_frame_(memory_frame),
	result_(false){
		tf_buffer_ = std::make_shared<tf2_ros::Buffer>();
		tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

		openvdb::initialize();
		attention_map_ = openvdb::FloatGrid::create();
		attention_map_->setTransform(openvdb::math::Transform::createLinearTransform(/*voxel size=*/0.1));

		std::string config_file;

		if (n.getParam("/rep_map/config_file", config_file)){
			try
			{
				YAML::Node config = YAML::LoadFile(config_file);
				std::vector<std::string> plugins_to_load = config["plugins"].as<std::vector<std::string>>();
				for (auto& plugin: plugins_to_load){
					loadPlugin(plugin);
				}
			}
			catch(YAML::BadFile& ex){
				ROS_WARN("The requested configuration %s was not found. The attention map manager will start without any plugin loaded.", config_file.c_str());
			}
		}else{
			ROS_WARN("Default plugins file parameter not found. The attention map manager will start without any plugin loaded.");
		}
	}

	void RepMapBase::compScreen(openvdb::FloatTree& attention_tree, openvdb::FloatTree& local_tree){
	    using Adapter = openvdb::TreeAdapter<openvdb::FloatTree>;
	    using TreeT = openvdb::FloatTree;
	    struct Local {
	        static inline void op(openvdb::CombineArgs<float>& args) {
	            args.setResult(1 - ((1-args.a())*(1-args.b())));
	        }
	    };
	    Adapter::tree(attention_tree).combineExtended(Adapter::tree(local_tree), Local::op, /*prune=*/false);
	}

	void RepMapBase::compIntComb(openvdb::Int32Tree& attention_tree, openvdb::Int32Tree& local_tree){
	    using Adapter = openvdb::TreeAdapter<openvdb::Int32Tree>;
	    using TreeT = openvdb::Int32Tree;
	    struct Local {
	        static inline void op(openvdb::CombineArgs<int32_t>& args) {
	            args.setResult(std::max(args.a(), args.b()));
	        }
	    };
	    Adapter::tree(attention_tree).combineExtended(Adapter::tree(local_tree), Local::op, /*prune=*/false);
	}

	void RepMapBase::compBooleanMask(openvdb::Int32Tree& tree_a,
										   openvdb::BoolTree& tree_b,
										   openvdb::Int32Tree& tree_ret){
	    struct Local {
	        static inline void op(openvdb::CombineArgs<int32_t, bool>& args) {
	            args.setResult(args.a());
	            args.setResultIsActive(args.b());
	        }
	    };
	    tree_ret.combine2Extended(tree_a, tree_b, Local::op, /*prune=*/false);
	}

	void RepMapBase::updateMap(){
		for (auto& plugin: plugins_){
			plugin.second->updateMap();
		}
	}

	void RepMapBase::updateGridResults(){
		local_grid_results_.clear();
		processed_grid_results_.clear();

		for (auto& plugin: plugins_){
			auto local_grid_results = plugin.second->getExtendedGrids();
			local_grid_results_.insert(local_grid_results_.end(), local_grid_results.begin(), local_grid_results.end());
		}

		// Starting a bunch of unifying processes for maps that might
		// be coming as partial from maps;
		std::vector<std::vector<std::string>> saliency_data_format = {{"hasSaliencyLowerThan"}, {"hasSaliencyHigherThan"}};
		processed_grid_results_.push_back(rep_plugins::GridResult(openvdb::FloatGrid::create(), saliency_data_format));
		for(auto& grid_result: local_grid_results_){
			if (grid_result.data_format_ ==  saliency_data_format){
				compScreen((openvdb::gridPtrCast<openvdb::FloatGrid>(processed_grid_results_.back().grid_ptr_))->tree(),
						   (openvdb::gridPtrCast<openvdb::FloatGrid>(grid_result.grid_ptr_))->tree());
				(openvdb::gridPtrCast<openvdb::FloatGrid>(processed_grid_results_.back().grid_ptr_))->setTransform((openvdb::gridPtrCast<openvdb::FloatGrid>(grid_result.grid_ptr_))->transformPtr());
			}
		}
		std::vector<std::string> global_saliency_data_format = {"as", "saliency"};
		processed_grid_results_.back().data_format_.push_back(global_saliency_data_format);
		saliency_data_format.push_back(global_saliency_data_format);
		// Unifying object detection algorithms
		std::vector<std::vector<std::string>> object_data_format = {{"containsObject"}};
		processed_grid_results_.push_back(rep_plugins::GridResult(openvdb::Int32Grid::create(), object_data_format));
		for(auto& grid_result: local_grid_results_){
			if (grid_result.data_format_ ==  object_data_format){
				compIntComb((openvdb::gridPtrCast<openvdb::Int32Grid>(processed_grid_results_.back().grid_ptr_))->tree(),
						   (openvdb::gridPtrCast<openvdb::Int32Grid>(grid_result.grid_ptr_))->tree());
				(openvdb::gridPtrCast<openvdb::Int32Grid>(processed_grid_results_.back().grid_ptr_))->setTransform((openvdb::gridPtrCast<openvdb::Int32Grid>(grid_result.grid_ptr_))->transformPtr());
			}
		}

		std::vector<std::string> global_objects_data_format = {"as", "object_class"};
		processed_grid_results_.back().data_format_.push_back(global_objects_data_format);
		object_data_format.push_back(global_objects_data_format);
		openvdb::Int32Grid::Ptr object_grid = nullptr;
		openvdb::FloatGrid::Ptr saliency_grid = nullptr;
		for(auto& processed_grid: processed_grid_results_){
			if(processed_grid.data_format_ == object_data_format){
				object_grid = openvdb::gridPtrCast<openvdb::Int32Grid>(processed_grid.grid_ptr_);
			}else if(processed_grid.data_format_ == saliency_data_format){
				saliency_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(processed_grid.grid_ptr_);
			}
		}

	}

	bool RepMapBase::loadPlugin(const std::string& plugin_name){
		bool result;

		try
		{
			auto find_it = plugins_.find(plugin_name);
			if (find_it != plugins_.end()){
				ROS_WARN("Pluging %s already loaded!", plugin_name.c_str());
				result = false;
			}else{
				AttentionPluginPtr plugin = plugin_loader_.createInstance(std::string("rep_plugins::")+plugin_name.c_str());
				plugins_[plugin_name] = plugin;
				local_maps_[plugin_name] = nullptr;
				plugin->updateFov(fov_);
				plugin->updateRefFrame(ref_frame_);
				plugin->updateFixedFrame(fixed_frame_);
				plugin->updateMemoryFrame(memory_frame_);
				plugin->initializeBase(n_, tf_buffer_);
				plugin->initialize();
				ROS_INFO("Plugin %s correctly loaded.", plugin_name.c_str());
				result = true;
			}
		}
		catch(pluginlib::PluginlibException& ex)
		{
			ROS_ERROR("The plugin failed to load. Error: %s", ex.what());
			result = false;
		}

		return result;
	}

	bool RepMapBase::unloadPlugin(std::string& plugin_name){
		bool result;

		try
		{
			auto find_it = plugins_.find(plugin_name);
			auto find_map_it = local_maps_.find(plugin_name);
			if (find_it == plugins_.end()){
				ROS_WARN("Plugin %s not found. Failed to unload.", plugin_name.c_str());
				result = false;
			}else{
				find_it->second.reset();
				plugins_.erase(find_it);
				ROS_INFO("Unloaded plugin %s", plugin_name.c_str());
				result = true;
			}
			if (find_map_it == local_maps_.end()){
				ROS_INFO("No local map belonging to this plugin was found");
			}else{
				local_maps_.erase(find_map_it);
				ROS_INFO("Removed local attention map belonging to the plugin");
			}
		}
		catch(pluginlib::PluginlibException& ex)
		{
			ROS_ERROR("Failed to unload the plugin. Error: %s", ex.what());
			result = false;
		}

		return result;
	}

	bool RepMapBase::addPlugin(rep_map::AddPlugin::Request& req, rep_map::AddPlugin::Response& res){
		res.result = loadPlugin(req.plugin_name); 
		return true;
	}

	bool RepMapBase::removePlugin(rep_map::RemovePlugin::Request& req, rep_map::RemovePlugin::Response& res){
		res.result = unloadPlugin(req.plugin_name);
		return true;
	}

	bool RepMapBase::processQuery(rep_map::Query::Request& req, rep_map::Query::Response& res){
		processed_query_ = parseQuery(req.query);
		std::string image_path = "/home/lorenzofe/Desktop/query.png";
		query_viz::generateImageFromWordVectors(processed_query_, query_image_pub);
		bool result = true;
		std::string searching_term;
		for (auto& line: processed_query_){
			if (searching_term.empty())
				searching_term = line[0];
			if (line[0] != searching_term)
				result = false;
			for (auto& word: line){
				std::cout<<word<<" ";
			}
			std::cout<<"\n";
		}
		std::cout<<"result: "<<result<<std::endl;
		res.result = result;
		result_ = result;
		return result;
	}

	void RepMapBase::performQuery(){
		if (result_){
			// pipelining
			// 1. from the first line extract the desired grid. Set it as a variant
			// 2. from now on, this is the input for all the other
			using grid_ptr_t = std::variant<openvdb::FloatGrid::Ptr, openvdb::Int32Grid::Ptr, openvdb::BoolGrid::Ptr>;
			bool initialized = false;
			grid_ptr_t queried_map; // Maybe this should be a deep copy...
			LocalMapOperation local_map_operation_type;
			bool set_threshold = false;
			for(auto& line: processed_query_){
				if (!initialized){
					std::vector<std::string> full_predicate = {line[1], line[2]};
					auto queried_map_base = findMapByPredicate(full_predicate);
					if (line[2] == "saliency"){
						queried_map = openvdb::gridPtrCast<openvdb::FloatGrid>(queried_map_base);
					}else if(line[2] == "object_class"){
						queried_map = openvdb::gridPtrCast<openvdb::Int32Grid>(queried_map_base);
					}else{
						break;
					}
					initialized = true;
				}else{
					std::string predicate = line[1];
					QueryFunction query_func;
					try{
						query_func = queries_map_.at(predicate);
					}catch(const std::out_of_range& e){
						std::cout<<"Query function handler not found. Trying to create one"<<std::endl;
						try{
							CompareOperator comp_op = predicate_operator_map.at(predicate);
							auto [query_it, insert] = queries_map_.insert(std::pair<std::string, QueryFunction>(predicate, comp_op));
							if (insert)
								query_func = query_it->second;
							else
								throw std::runtime_error("error");
						}catch(const std::out_of_range& ex){
							//Now we start looking among the local grids
							LocalMapOperation local_operation;
							try{
								local_operation = local_map_operations.at(predicate);
								std::cout<<"Local operation found!"<<std::endl;
								try{
									auto operation_implementation = local_operation_implementation.at(local_operation);
									local_map_operation_type = local_operation;
									auto [query_it, insert] = queries_map_.insert(std::pair<std::string, QueryFunction>(predicate, operation_implementation.first));
									if (insert){
										query_it->second.setThreshold(operation_implementation.second);
										query_it->second.setConstantThreshold();
										query_func = query_it->second;
										set_threshold = true;
									}
									else{
										throw std::runtime_error("error");
									}
								}catch(const std::out_of_range& ex){
									std::cout<<"We did not found a way to implement a query; aborting the line"<<std::endl;
									continue;
								}catch(const std::runtime_error& ex){
									std::cout<<"We did not found a way to implement a query; aborting the line"<<std::endl;
									continue;
								}
							}catch(const std::out_of_range& ex){
								std::cout<<"We did not found a way to implement a query; aborting the line"<<std::endl;
								continue;
							}
						}catch(const std::runtime_error& ex){
							std::cout<<"We did not found a way to implement a query; aborting the line"<<std::endl;
							continue;
						}
					}
					// At this point, query_func should be set up. Now we look for the second term map
					grid_ptr_t second_term_map;
					std::vector<std::string> second_term_data_format;
					try{
						second_term_data_format = predicate_second_term_map.at(predicate);
					}catch(const std::out_of_range& e){
						second_term_data_format = {};
					}
					openvdb::GridBase::Ptr second_term_base = findMapByPredicate(second_term_data_format);
					if (second_term_base && (second_term_data_format.size()==2)&&(second_term_data_format[1]=="saliency")){
						second_term_map = openvdb::gridPtrCast<openvdb::FloatGrid>(second_term_base);
					}else if (second_term_base && (second_term_data_format.size()==2)&&(second_term_data_format[1]=="object_class")){
						second_term_map = openvdb::gridPtrCast<openvdb::Int32Grid>(second_term_base);
					}else{
						// We might be looking at a local map function
						// We will look at the exact matching between the data line[1] and line[2] values and the data format
						std::vector<std::string> full_predicate_term = {line[1], line[2]};
						bool found = false;
						second_term_base = findLocalMapByPredicate(full_predicate_term);
						if (!second_term_base){
							std::cout<<"Not able to find the correspondant second term; aborting"<<std::endl;
							continue;
						}else{
							try{
								local_map_operation_type = local_map_operations.at(line[1]);
							}catch(const std::out_of_range& e){
								std::cout<<"Unknown local map operation"<<std::endl;
								continue;
							}
							switch(local_map_operation_type){
								case LocalMapOperation::BOOL_MASK:
									second_term_map = openvdb::gridPtrCast<openvdb::BoolGrid>(second_term_base);
									std::cout<<"Found a local mapppp"<<std::endl;
									break;
							}
						}
					}
					// Now we create the return map. This is a deep copy of the first map.
					grid_ptr_t next_step_map;
					if (std::holds_alternative<openvdb::FloatGrid::Ptr>(queried_map)){
						auto temp_queried_map = std::get<openvdb::FloatGrid::Ptr>(queried_map);
						next_step_map = openvdb::gridPtrCast<openvdb::FloatGrid>(temp_queried_map->deepCopyGrid());
					}else if (std::holds_alternative<openvdb::Int32Grid::Ptr>(queried_map)){
						auto temp_queried_map = std::get<openvdb::Int32Grid::Ptr>(queried_map);
						next_step_map = openvdb::gridPtrCast<openvdb::Int32Grid>(temp_queried_map->deepCopyGrid());
					}else{
						continue; // Need something for boolean masks
					}
					// At this point we need to get the value for the query
					std::string threshold = line[2];
					if (!query_func.isConstantThreshold()){
						if (checkThresholdFormat<int>(threshold)){
							query_func.setThreshold(std::stoi(threshold));
						}else if(checkThresholdFormat<float>(threshold)){
							query_func.setThreshold(std::stof(threshold));
						}else{
							std::cout<<"I was not able to transform infer the threshold value type; continue"<<std::endl;
						}
					}
					// Setting up the trees
					std::cout<<predicate<<std::endl;
					//query_func.setThreshold(true);
					query_func.setGridA(queried_map);
					query_func.setGridB(second_term_map);
					query_func.setGridRet(next_step_map);
					query_func.performQuery();
					queried_map = next_step_map;
				}
				queried_map_ = queried_map;
			}
		}
	}
}