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
#pragma once

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sophus/se3.hpp>

struct SE3Pose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SE3Pose(const double time, const Sophus::SE3d &Twc)
        : time_(time)
    {
        twc_[0] = Twc.translation().x(); 
        twc_[1] = Twc.translation().y(); 
        twc_[2] = Twc.translation().z();
        qwc_[0] = Twc.so3().unit_quaternion().x(); 
        qwc_[1] = Twc.so3().unit_quaternion().y(); 
        qwc_[2] = Twc.so3().unit_quaternion().z();
        qwc_[3] = Twc.so3().unit_quaternion().w(); 
    }

    double time_;
    double twc_[3];
    double qwc_[4];
};

struct FramePose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FramePose(const Sophus::SE3d &Twc, const bool iskf)
        : iskf_(iskf)
    {
        twc_[0] = Twc.translation().x(); 
        twc_[1] = Twc.translation().y(); 
        twc_[2] = Twc.translation().z();
        qwc_[0] = Twc.so3().unit_quaternion().x(); 
        qwc_[1] = Twc.so3().unit_quaternion().y(); 
        qwc_[2] = Twc.so3().unit_quaternion().z();
        qwc_[3] = Twc.so3().unit_quaternion().w(); 
    }

    bool iskf_;

    double twc_[3];
    double qwc_[4];

    double tprev_cur_[3] = {0., 0., 0.};
    double qprev_cur_[4] = {0., 0., 0., 1.};
};

struct KittiPose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KittiPose(const Sophus::SE3d &Twc)
    {
        Eigen::Vector3d twc = Twc.translation();
        Eigen::Matrix3d Rwc = Twc.rotationMatrix();
        Twc_[3] = twc.x();
        Twc_[7] = twc.y(); 
        Twc_[11] = twc.z();

        for( int i = 0 ; i < 3 ; i++ ) {
            Twc_[i] = Rwc(0,i);
            Twc_[i+4] = Rwc(1,i);
            Twc_[i+8] = Rwc(2,i);
        }
    }

    double Twc_[12];
};

class Logger {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static void addSE3Pose(const double time, const Sophus::SE3d &Twc, const bool iskf) {
        vse3pose_.push_back(SE3Pose(time, Twc));
        vfullse3pose_.push_back(SE3Pose(time, Twc));

        vkittipose_.push_back(KittiPose(Twc));

        if( vframepose_.empty() ) {
            vframepose_.push_back(FramePose(Twc, iskf));
        } else {
            FramePose prevpose = vframepose_.back();
            Eigen::Map<Eigen::Vector3d> t(prevpose.twc_);
            Eigen::Map<Eigen::Quaterniond> q(prevpose.qwc_);
            Sophus::SE3d Twprev(q,t);
            Sophus::SE3d Tprev_cur = Twprev.inverse() * Twc;

            FramePose curpose(Twc, iskf);

            Eigen::Map<Eigen::Vector3d> curt(curpose.tprev_cur_);
            Eigen::Map<Eigen::Quaterniond> curq(curpose.qprev_cur_);

            curt = Tprev_cur.translation();
            curq = Tprev_cur.unit_quaternion();
            
            vframepose_.push_back(curpose);
        }
    }

    static void addKfSE3Pose(const double time, const Sophus::SE3d &Twc) {
        vse3kfpose_.emplace(time, SE3Pose(time, Twc));
    }

    static void writeTrajectory(const std::string &filename) {

        std::ofstream f;

        std::cout << "\n Going to write the computed trajectory into : " << filename << "\n";

        f.open(filename.c_str());
        f << std::fixed;
        
        size_t nbposes = vse3pose_.size();
        for( size_t i = 0 ; i < nbposes ; i++ )
        {
            double &time = vse3pose_[i].time_;
            double *t = vse3pose_[i].twc_;
            double *q = vse3pose_[i].qwc_;
            
            f << time << " " << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] 
                << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;

            f.flush();
        }

        f.close();

        std::cout << "\nTrajectory file written!\n";
    }

    static void writeTrajectoryTartanAir(const std::string &filename) {

        std::ofstream f;

        std::cout << "\n Going to write the computed trajectory into : " << filename << "\n";

        f.open(filename.c_str());
        f << std::fixed;
        
        size_t nbposes = vfullse3pose_.size();
        for( size_t i = 0 ; i < nbposes ; i++ )
        {
            double *t = vfullse3pose_[i].twc_;
            double *q = vfullse3pose_[i].qwc_;
            
            f << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] 
                << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;

            f.flush();
        }

        f.close();

        std::cout << "\nTartan Air Trajectory file written!\n";
    }

    static void writeTrajectoryKITTI(const std::string &filename) {

        std::ofstream f;

        std::cout << "\n Going to write the computed trajectory into : " << filename << "\n";

        f.open(filename.c_str());
        f << std::fixed;
        
        size_t nbposes = vkittipose_.size();
        for( size_t i = 0 ; i < nbposes ; i++ )
        {
            double *T = vkittipose_.at(i).Twc_;
            
            f << std::setprecision(9) 
                << T[0] << " " << T[1] << " " << T[2] << " " << T[3] << " "
                << T[4] << " " << T[5] << " " << T[6] << " " << T[7] << " "
                << T[8] << " " << T[9] << " " << T[10] << " " << T[11]
                << std::endl;

            f.flush();
        }

        f.close();

        std::cout << "\nKITTITrajectory file written!\n";
    }

    static void writeKfsTrajectory(const std::string &filename) {

        std::ofstream f;

        std::cout << "\n Going to write the computed KFs trajectory into : " << filename << "\n";

        f.open(filename.c_str());
        f << std::fixed;
        
        for( auto & se3kfpose : vse3kfpose_ )
        {
            double time = se3kfpose.first;
            double *t = se3kfpose.second.twc_;
            double *q = se3kfpose.second.qwc_;
            
            f << time << " " << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2]
                << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;

            f.flush();
        }

        f.close();

        std::cout << "\nKfs Trajectory file written!\n";
    }

    static void writeKfsTrajectoryTartanAir(const std::string &filename) {

        std::ofstream f;

        std::cout << "\n Going to write the computed trajectory into : " << filename << "\n";

        f.open(filename.c_str());
        f << std::fixed;

        size_t nbposesmissing = vfullse3pose_.size() - vse3kfpose_.size();

        for( size_t i = 0 ; i < nbposesmissing ; i++ ) {
            f << std::setprecision(9) << 0. << " " << 0. << " " << 0.
                << " " << 0. << " " << 0. << " " << 0. << " " << 1. << std::endl;
        }

        for( auto & se3kfpose : vse3kfpose_ )
        {
            double *t = se3kfpose.second.twc_;
            double *q = se3kfpose.second.qwc_;
            
            f << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2]
                << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;

            f.flush();
        }

        f.close();

        std::cout << "\nTartan Air Trajectory file written!\n";
    }

    static void reset() {
        vse3pose_.clear();
        vse3kfpose_.clear();
        for( auto &el : vfullse3pose_ ) {
            el = SE3Pose(el.time_, Sophus::SE3d());
        }
    }

    static std::vector<SE3Pose> vse3pose_, vfullse3pose_;
    static std::map<double, SE3Pose> vse3kfpose_;


    static std::vector<KittiPose> vkittipose_;

    // For outputting full traj while taking into account
    // loop-closures in the traj!
    static std::vector<FramePose> vframepose_;
};

// Has to be defined before being used
std::vector<SE3Pose> Logger::vse3pose_, Logger::vfullse3pose_;
std::map<double, SE3Pose>  Logger::vse3kfpose_;
std::vector<KittiPose> Logger::vkittipose_;
std::vector<FramePose> Logger::vframepose_;