#!/bin/bash

parameters=("euroc_stereo") 

for param in "${parameters[@]}"
do
    for i in {1..5}
    do
        for j in {1..5}
        do
            echo "$param.yaml";
            rosrun ov2slam ov2slam_node /home/maxime/catkin_ws/src/ov2slam/parameters_files/$param.yaml > log.txt &
            sleep 3 && rosbag play /home/maxime/Documents/these/datasets/EuRoC_MAV/MH_0$i\_*.bag -r 1. &

            wait;

            sleep 1;

            mv ov2slam_traj.txt ov2slam_traj_mh0$i\_$param\_$j.txt;
            mv ov2slam_kfs_traj.txt ov2slam_kfs_traj_mh0$i\_$param\_$j.txt;
            mv ov2slam_full_traj_wlc_opt.txt ov2slam_full_traj_wlc_mh0$i\_$param\_$j.txt;
            
            sleep 1;

        done;
    done;
done;