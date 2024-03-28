# rep_map
Utilities for combining and querying Representation Maps

> [!WARNING]  
> This is a research project and an official release is still
> under development.

`rep_map` presents the required interfaces for combining and querying
`RMs`. It provides a `ROS` node (`nodes/rep_map`) managing:
- the `plugins` loading and unloading procedures;
- the combination of the `RMs` generated by each plugin;
- the querying procedures;

## Dependencies
- `ROS noetic`
- `openvdb >= 6.2.1` 
- `tbb >= 2020.1`
- `pcl >= 1.10.0`
- `yaml-cpp >= 0.6.2`
- `boost >= 1.71.0`
