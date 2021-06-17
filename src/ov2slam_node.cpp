/**
*    This file is part of OV²SLAM.
*    
*    Copyright (C) 2020 ONERA
*
*    For more information see <https://github.com/ov2slam/ov2slam>
*
*    OV²SLAM is free software: you can redistribute it and/or modify
*    it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    OV²SLAM is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU General Public License for more details.
*
*    You should have received a copy of the GNU General Public License
*    along with OV²SLAM.  If not, see <https://www.gnu.org/licenses/>.
*
*    Authors: Maxime Ferrera     <maxime.ferrera at gmail dot com> (ONERA, DTIS - IVA),
*             Alexandre Eudes    <first.last at onera dot fr>      (ONERA, DTIS - IVA),
*             Julien Moras       <first.last at onera dot fr>      (ONERA, DTIS - IVA),
*             Martial Sanfourche <first.last at onera dot fr>      (ONERA, DTIS - IVA)
*/

#include <experimental/filesystem>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include <ros/console.h>
#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>

#include "ov2slam.hpp"
#include "slam_params.hpp"

class SensorsGrabber {

public:
    SensorsGrabber(SlamManager* slam)
        : pslam_(slam)
    {
        std::cout << "\nSensors Grabber is created...\n";
    }

    void subLeftImage(const sensor_msgs::ImageConstPtr& image)
    {
        std::lock_guard<std::mutex> lock(img_mutex);
        img0_buf.push(image);
    }

    void subRightImage(const sensor_msgs::ImageConstPtr& image)
    {
        std::lock_guard<std::mutex> lock(img_mutex);
        img1_buf.push(image);
    }

    cv::Mat getGrayImageFromMsg(const sensor_msgs::ImageConstPtr& img_msg)
    {
        // Get and prepare images
        cv_bridge::CvImageConstPtr ptr;
        try {
            ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("\n\n\ncv_bridge exeception: %s\n\n\n", e.what());
        }

        return ptr->image;
    }

    // extract images with same timestamp from two topics
    // (mostly derived from Vins-Fusion: https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)
    void sync_process()
    {
        std::cout << "\nStarting the measurements reader thread!\n";

        while (!pslam_->bexit_required_) {
            if (pslam_->pslamstate_->stereo_) {
                cv::Mat image0, image1;

                std::lock_guard<std::mutex> lock(img_mutex);

                if (!img0_buf.empty() && !img1_buf.empty()) {
                    double time0 = img0_buf.front()->header.stamp.toSec();
                    double time1 = img1_buf.front()->header.stamp.toSec();

                    // sync tolerance
                    if (time0 < time1 - 0.015) {
                        img0_buf.pop();
                        std::cout << "\n Throw img0 -- Sync error : " << (time0 - time1) << "\n";
                    } else if (time0 > time1 + 0.015) {
                        img1_buf.pop();
                        std::cout << "\n Throw img1 -- Sync error : " << (time0 - time1) << "\n";
                    } else {
                        image0 = getGrayImageFromMsg(img0_buf.front());
                        image1 = getGrayImageFromMsg(img1_buf.front());
                        img0_buf.pop();
                        img1_buf.pop();

                        if (!image0.empty() && !image1.empty()) {
                            pslam_->addNewStereoImages(time0, image0, image1);
                        }
                    }
                }
            } else if (pslam_->pslamstate_->mono_) {
                cv::Mat image0;

                std::lock_guard<std::mutex> lock(img_mutex);

                if (!img0_buf.empty()) {
                    double time = img0_buf.front()->header.stamp.toSec();
                    image0 = getGrayImageFromMsg(img0_buf.front());
                    img0_buf.pop();

                    if (!image0.empty()) {
                        pslam_->addNewMonoImage(time, image0);
                    }
                }
            }

            std::chrono::milliseconds dura(1);
            std::this_thread::sleep_for(dura);
        }

        std::cout << "\n Bag reader SyncProcess thread is terminating!\n";
    }

    std::queue<sensor_msgs::ImageConstPtr> img0_buf;
    std::queue<sensor_msgs::ImageConstPtr> img1_buf;
    std::mutex img_mutex;

    SlamManager* pslam_;
};

// read images from folder and make sure they are sorted by filenames
bool read_images_from_folder(std::string folder, float fps, std::vector<std::string>& filenames, std::vector<float>& timestamps)
{
    if (!std::experimental::filesystem::exists(folder))
        return false;
    std::vector<std::string> vimage_path;
    for (const auto& entry : std::experimental::filesystem::directory_iterator(folder)) {
        vimage_path.push_back(entry.path().string());
    }

    std::vector<std::string> vimage_extension(vimage_path.size());
    std::vector<std::string> vimg_name, vimg_name_sorted;
    std::vector<int> vimg_name_int;
    for (size_t j = 0; j < vimage_path.size(); j++) {
        std::string image_path = std::string(vimage_path[j].c_str());
        int loc1 = image_path.find_last_of("\\/") + 1;
        int loc2 = image_path.find_last_of(".");
        std::string image_name = image_path.substr(loc1, loc2 - loc1);
        std::string image_ext = image_path.substr(loc2 + 1, image_path.size());
        vimg_name.push_back(image_name);
        vimg_name_int.push_back(std::stoi(image_name));
        vimage_extension[j] = image_ext;
    }
    std::vector<std::size_t> index(vimg_name.size());
    std::iota(index.begin(), index.end(), 0);
    std::sort(index.begin(), index.end(), [&](size_t a, size_t b) { return vimg_name_int[a] < vimg_name_int[b]; });

    for (auto i : index) {
        vimg_name_sorted.push_back(vimg_name[i]);
    }
    filenames.resize(vimg_name_sorted.size());
    timestamps.resize(vimg_name_sorted.size());

    for (size_t i = 0; i < vimg_name_sorted.size(); i++) {
        filenames[i] = folder + "/" + vimg_name_sorted[i] + "." + vimage_extension[i];
        timestamps[i] = (float)i / fps;
    }
    return true;
}
int main(int argc, char** argv)
{
    // Init the node
    ros::init(argc, argv, "ov2slam_node");

    if (argc < 2) {
        std::cout << "\nUsage: rosrun ov2slam ov2slam_node parameters_files/params.yaml\n";
        return 1;
    }

    std::cout << "\nLaunching OV²SLAM...\n\n";

    ros::NodeHandle nh("~");

    // Load the parameters
    std::string parameters_file = argv[1];

    std::cout << "\nLoading parameters file : " << parameters_file << "...\n";

    const cv::FileStorage fsSettings(parameters_file.c_str(), cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        std::cout << "Failed to open settings file...";
        return 1;
    } else {
        std::cout << "\nParameters file loaded...\n";
    }

    std::shared_ptr<SlamParams> pparams;
    pparams.reset(new SlamParams(fsSettings));

    // Create the ROS Visualizer
    std::shared_ptr<RosVisualizer> prosviz;
    prosviz.reset(new RosVisualizer(nh));

    // Setting up the SLAM Manager
    SlamManager slam(pparams, prosviz);

    if (pparams->bread_images_from_folder_) {
        std::vector<std::string> lfilenames, rfilenames;
        std::vector<float> timestamps;

        if (!read_images_from_folder(fsSettings["Camera.topic_left"], fsSettings["fps"], lfilenames, timestamps)) {
            std::cout << "Failed to open image folder...";
            return 1;
        }
        if (pparams->stereo_) {
            if (!read_images_from_folder(fsSettings["Camera.topic_right"], fsSettings["fps"], rfilenames, timestamps)) {
                std::cout << "Failed to open image folder...";
                return 1;
            }
        }
        slam.setFilenamesTimestamps(lfilenames, rfilenames, timestamps);

        // Start the SLAM thread
        std::thread slamthread(&SlamManager::run, &slam);

        // ROS Spin
        ros::spin();

        // Request Slam Manager thread to exit
        slam.bexit_required_ = true;

        // Waiting end of SLAM Manager
        while (slam.bis_on_) {
            std::chrono::seconds dura(1);
            std::this_thread::sleep_for(dura);
        }
    }

    if (!pparams->bread_images_from_folder_) {
        // Start the SLAM thread
        std::thread slamthread(&SlamManager::run, &slam);

        // Create the Bag file reader & callback functions
        SensorsGrabber sb(&slam);

        // Create callbacks according to the topics set in the parameters file
        ros::Subscriber subleft = nh.subscribe(fsSettings["Camera.topic_left"], 2, &SensorsGrabber::subLeftImage, &sb);
        ros::Subscriber subright = nh.subscribe(fsSettings["Camera.topic_right"], 2, &SensorsGrabber::subRightImage, &sb);

        // Start a thread for providing new measurements to the SLAM
        std::thread sync_thread(&SensorsGrabber::sync_process, &sb);

        // ROS Spin
        ros::spin();

        // Request Slam Manager thread to exit
        slam.bexit_required_ = true;

        // Waiting end of SLAM Manager
        while (slam.bis_on_) {
            std::chrono::seconds dura(1);
            std::this_thread::sleep_for(dura);
        }
    }

    return 0;
}
