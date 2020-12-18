#!/bin/bash

parameters=("euroc_stereo") 

for param in "${parameters[@]}"
do
    for i in {1..5}
    do
        for j in {1..5}
        do
            echo "$param.yaml";
            rosrun onera_slam onera_slam_node /home/maxime/catkin_ws/src/onera_slam/parameters_files/$param.yaml > log.txt &
            sleep 3 && rosbag play /home/maxime/Documents/these/datasets/EuRoC_MAV/MH_0$i\_*.bag -r 1. &

            wait;

            sleep 1;

            python /home/maxime/softwares/evo/evo/main_ape_multiple_trajs.py tum ~/Documents/these/datasets/EuRoC_MAV/groundtruth_files/mh0$i\_gt.tum onera_slam_traj.txt -a ; 

            mv onera_slam_traj.txt onera_slam_traj_mh0$i\_$param\_$j.txt;
            mv evo_ape_stats.txt onera_slam_stats_mh0$i\_$param\_$j.txt;

            python /home/maxime/softwares/evo/evo/main_ape_multiple_trajs.py tum ~/Documents/these/datasets/EuRoC_MAV/groundtruth_files/mh0$i\_gt.tum onera_slam_kfs_traj.txt -a ;  
            
            mv onera_slam_kfs_traj.txt onera_slam_kfs_traj_mh0$i\_$param\_$j.txt;
            mv evo_ape_stats.txt onera_slam_kfs_stats_mh0$i\_$param\_$j.txt;
            
            python /home/maxime/softwares/evo/evo/main_ape_multiple_trajs.py tum ~/Documents/these/datasets/EuRoC_MAV/groundtruth_files/mh0$i\_gt.tum onera_slam_fullba_kfs_traj.txt -a ;

            mv onera_slam_fullba_kfs_traj.txt onera_slam_fullba_kfs_traj_mh0$i\_$param\_$j.txt;
            mv evo_ape_stats.txt onera_slam_fullba_kfs_stats_mh0$i\_$param\_$j.txt;

        done;
    done;
done;

for param in "${parameters[@]}"
do
    for k in {1..2}
    do
        for i in {1..3}
        do
            for j in {1..5}
            do
                echo "$param.yaml";
                rosrun onera_slam onera_slam_node /home/maxime/catkin_ws/src/onera_slam/parameters_files/$param.yaml > log.txt &
                sleep 3 && rosbag play /home/maxime/Documents/these/datasets/EuRoC_MAV/V$k\_0$i\_*.bag -r 1. &

                wait;

                sleep 1;

                python /home/maxime/softwares/evo/evo/main_ape_multiple_trajs.py tum ~/Documents/these/datasets/EuRoC_MAV/groundtruth_files/v0$k\_0$i\_gt.tum onera_slam_traj.txt -a ; 

                mv onera_slam_traj.txt onera_slam_traj_v$k\_0$i\_$param\_$j.txt;
                mv evo_ape_stats.txt onera_slam_stats_v$k\_0$i\_$param\_$j.txt;

                python /home/maxime/softwares/evo/evo/main_ape_multiple_trajs.py tum ~/Documents/these/datasets/EuRoC_MAV/groundtruth_files/v0$k\_0$i\_gt.tum onera_slam_kfs_traj.txt -a ; 

                mv onera_slam_kfs_traj.txt onera_slam_kfs_traj_v$k\_0$i\_$param\_$j.txt;
                mv evo_ape_stats.txt onera_slam_kfs_stats_v$k\_0$i\_$param\_$j.txt;

                python /home/maxime/softwares/evo/evo/main_ape_multiple_trajs.py tum ~/Documents/these/datasets/EuRoC_MAV/groundtruth_files/v0$k\_0$i\_gt.tum onera_slam_fullba_kfs_traj.txt -a ;

                mv onera_slam_fullba_kfs_traj.txt onera_slam_fullba_kfs_traj_v$k\_0$i\_$param\_$j.txt;
                mv evo_ape_stats.txt onera_slam_fullba_kfs_stats_v$k\_0$i\_$param\_$j.txt;

            done
        done
    done
done