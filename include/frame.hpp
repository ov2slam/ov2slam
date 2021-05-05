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


#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>

#include <sophus/se3.hpp>

#include "camera_calibration.hpp"

struct Keypoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int lmid_;

    cv::Point2f px_;
    cv::Point2f unpx_;
    Eigen::Vector3d bv_;

    int scale_;
    float angle_;
    cv::Mat desc_;
    
    bool is3d_;

    bool is_stereo_;
    cv::Point2f rpx_;
    cv::Point2f runpx_;
    Eigen::Vector3d rbv_;

    bool is_retracked_;

    Keypoint() : lmid_(-1), scale_(0), angle_(-1.), is3d_(false), is_stereo_(false), is_retracked_(false)
    {}

    // For using kps in ordered containers
    bool operator< (const Keypoint &kp) const
    {
        return lmid_ < kp.lmid_;
    }
};

class Frame {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Frame();
    Frame(std::shared_ptr<CameraCalibration> pcalib_left, const size_t ncellsize);
    Frame(std::shared_ptr<CameraCalibration> pcalib_left, std::shared_ptr<CameraCalibration> pcalib_right, const size_t ncellsize);
    Frame(const Frame &F);

    void updateFrame(const int id, const double img_time);

    std::vector<Keypoint> getKeypoints() const;
    std::vector<Keypoint> getKeypoints2d() const;
    std::vector<Keypoint> getKeypoints3d() const;
    std::vector<Keypoint> getKeypointsStereo() const;
    
    std::vector<cv::Point2f> getKeypointsPx() const;
    std::vector<cv::Point2f> getKeypointsUnPx() const;
    std::vector<Eigen::Vector3d> getKeypointsBv() const;
    std::vector<int> getKeypointsId() const;
    std::vector<cv::Mat> getKeypointsDesc() const;

    Keypoint getKeypointById(const int lmid) const;

    std::vector<Keypoint> getKeypointsByIds(const std::vector<int> &vlmids) const;

    void computeKeypoint(const cv::Point2f &pt, Keypoint &kp);
    Keypoint computeKeypoint(const cv::Point2f &pt, const int lmid);

    void addKeypoint(const Keypoint &kp);
    void addKeypoint(const cv::Point2f &pt, const int lmid);
    void addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc);
    void addKeypoint(const cv::Point2f &pt, const int lmid, const int scale);
    void addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc, const int scale);
    void addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc, const int scale, const float angle);

    void updateKeypoint(const cv::Point2f &pt, Keypoint &kp);
    void updateKeypoint(const int lmid, const cv::Point2f &pt);
    void updateKeypointDesc(const int lmid, const cv::Mat &desc);
    void updateKeypointAngle(const int lmid, const float angle);

    bool updateKeypointId(const int prevlmid, const int newlmid, const bool is3d);

    void computeStereoKeypoint(const cv::Point2f &pt, Keypoint &kp);
    void updateKeypointStereo(const int lmid, const cv::Point2f &pt);

    void removeKeypoint(const Keypoint &kp);
    void removeKeypointById(const int lmid);

    void removeStereoKeypoint(const Keypoint &kp);
    void removeStereoKeypointById(const int lmid);

    void addKeypointToGrid(const Keypoint &kp);
    void removeKeypointFromGrid(const Keypoint &kp);
    void updateKeypointInGrid(const Keypoint &prevkp, const Keypoint &newkp);
    std::vector<Keypoint> getKeypointsFromGrid(const cv::Point2f &pt) const;
    int getKeypointCellIdx(const cv::Point2f &pt) const;

    std::vector<Keypoint> getSurroundingKeypoints(const Keypoint &kp) const;
    std::vector<Keypoint> getSurroundingKeypoints(const cv::Point2f &pt) const;

    void turnKeypoint3d(const int lmid);

    bool isObservingKp(const int lmid) const;

    Sophus::SE3d getTcw() const;
    Sophus::SE3d getTwc() const;

    Eigen::Matrix3d getRcw() const;
    Eigen::Matrix3d getRwc() const;

    Eigen::Vector3d gettcw() const;
    Eigen::Vector3d gettwc() const;

    void setTwc(const Sophus::SE3d &Twc);
    void setTcw(const Sophus::SE3d &Tcw);

    void setTwc(const Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc);
    void setTcw(const Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw);

    std::set<int> getCovisibleKfSet() const;

    std::map<int,int> getCovisibleKfMap() const;
    void updateCovisibleKfMap(const std::map<int,int> &cokfs);
    void addCovisibleKf(const int kfid);
    void removeCovisibleKf(const int kfid);
    void decreaseCovisibleKf(const int kfid);

    cv::Point2f projCamToImageDist(const Eigen::Vector3d &pt) const;
    cv::Point2f projCamToImage(const Eigen::Vector3d &pt) const;

    cv::Point2f projCamToRightImageDist(const Eigen::Vector3d &pt) const;
    cv::Point2f projCamToRightImage(const Eigen::Vector3d &pt) const;

    cv::Point2f projDistCamToImage(const Eigen::Vector3d &pt) const;
    cv::Point2f projDistCamToRightImage(const Eigen::Vector3d &pt) const;

    Eigen::Vector3d projCamToWorld(const Eigen::Vector3d &pt) const;
    Eigen::Vector3d projWorldToCam(const Eigen::Vector3d &pt) const;

    cv::Point2f projWorldToImage(const Eigen::Vector3d &pt) const;
    cv::Point2f projWorldToImageDist(const Eigen::Vector3d &pt) const;

    cv::Point2f projWorldToRightImage(const Eigen::Vector3d &pt) const;
    cv::Point2f projWorldToRightImageDist(const Eigen::Vector3d &pt) const;

    bool isInImage(const cv::Point2f &pt) const;
    bool isInRightImage(const cv::Point2f &pt) const;

    void displayFrameInfo();

    // For using frame in ordered containers
    bool operator< (const Frame &f) const {
        return id_ < f.id_;
    }

    void reset();

    // Frame info
    int id_, kfid_;
    double img_time_;

    // Hash Map of observed keypoints
    std::unordered_map<int, Keypoint> mapkps_;

    // Grid of kps sorted by cell numbers and scale
    // (We use const pointer to reference the keypoints in vkps_
    // HENCE we should only use the grid to read kps)
    std::vector<std::vector<int>> vgridkps_;
    size_t ngridcells_, noccupcells_, ncellsize_, nbwcells_, nbhcells_;

    size_t nbkps_, nb2dkps_, nb3dkps_, nb_stereo_kps_;

    // Pose (T cam -> world), (T world -> cam)
    Sophus::SE3d Twc_, Tcw_;

    /* TODO
    Set a vector of calib ptrs to handle any multicam system.
    Each calib ptr should contain an extrinsic parametrization with a common
    reference frame. If cam0 is the ref., its extrinsic would be the identity.
    Would mean an easy integration of IMU body frame as well.
    */
    // Calibration model
    std::shared_ptr<CameraCalibration> pcalib_leftcam_;
    std::shared_ptr<CameraCalibration> pcalib_rightcam_;

    Eigen::Matrix3d Frl_;
    cv::Mat Fcv_;

    // Covisible kf ids
    std::map<int,int> map_covkfs_;

    // Local MapPoint ids
    std::unordered_set<int> set_local_mapids_;

    // Mutex
    mutable std::mutex kps_mutex_, pose_mutex_;
    mutable std::mutex grid_mutex_, cokfs_mutex_;
};
