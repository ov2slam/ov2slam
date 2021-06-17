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
#include <string>
#include <mutex>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include <sophus/se3.hpp>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

class CameraCalibration {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum Model {
        Pinhole,
        Fisheye
    };

    CameraCalibration() {}

    CameraCalibration(const std::string &model, double fx, double fy, double cx, double cy,
        double k1, double k2, double p1, double p2, double img_w, double img_h);

    void setUndistMap(const double alpha);

    void setUndistStereoMap(const cv::Mat &R, const cv::Mat &P, const cv::Rect &roi);

    void setupExtrinsic(const Sophus::SE3d &Tc0ci);

    void setROIMask(const cv::Rect &roi);

    void rectifyImage(const cv::Mat &img, cv::Mat &rect) const;

    cv::Point2f projectCamToImageDist(const Eigen::Vector3d &pt) const;
    cv::Point2f projectCamToImageUndist(const Eigen::Vector3d &pt) const;

    cv::Point2f projectCamToImage(const Eigen::Vector3d &pt) const;
    cv::Point2f undistortImagePoint(const cv::Point2f &pt) const;

    Eigen::Matrix3d getRotation() const;
    Eigen::Vector3d getTranslation() const;
    Sophus::SE3d getExtrinsic() const;

    void updateExtrinsic(const Sophus::SE3d &Tc0ci);
    void updateIntrinsic(const double fx, const double fy, const double cx, const double cy);
    void updateDistCoefs(const double k1, const double k2=0., const double p1=0., const double p2=0.);

    void displayCalib() {
        std::cout.precision(8);
        std::cout << "\n fx : " << fx_ << " - fy : " << fy_ << " - cx : " << cx_ << " - cy : " << cy_ << "\n";
    }

    void displayDist() {
        std::cout.precision(8);
        std::cout << "\n k1 : " << k1_ << " - k2 : " << k2_ << " - p1 : " << p1_ << " - p2 : " << p2_ << "\n";
    }
    
    // Model enum
    Model model_;

    // Calibration model
    double fx_, fy_, cx_, cy_;
    double k1_, k2_, p1_, p2_;
    cv::Mat Kcv_, Dcv_;
    Eigen::Matrix3d K_;
    Eigen::Vector4d D_;

    Eigen::Matrix3d iK_;
    double ifx_, ify_, icx_, icy_;

    // Image size
    double img_w_, img_h_;
    cv::Size img_size_;

    // Extrinsic Parameters (This cam to cam 0)
    Sophus::SE3d Tc0ci_, Tcic0_;
    cv::Mat Rcv_c0ci_, tcv_c0ci_;
    cv::Mat Rcv_cic0_, tcv_cic0_;

    // Undistort Maps
    cv::Mat undist_map_x_, undist_map_y_;

    Eigen::Matrix3d Rrectraw_;

    mutable std::mutex intrinsic_mutex_, extrinsic_mutex_;

    // ROI Mask for detection
    cv::Rect roi_rect_;
    cv::Mat roi_mask_;
};