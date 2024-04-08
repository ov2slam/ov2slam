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


#include <mutex>
#include <unordered_map>

#include <pcl_conversions/pcl_conversions.h>

#include "slam_params.hpp"
#include "frame.hpp"
#include "map_point.hpp"
#include "feature_extractor.hpp"
#include "feature_tracker.hpp"


class MapManager {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MapManager() {}

    MapManager(std::shared_ptr<SlamParams> pstate, std::shared_ptr<Frame> pframe, std::shared_ptr<FeatureExtractor> pfeatextract, std::shared_ptr<FeatureTracker> ptracker);

    void prepareFrame();
    
    void addKeyframe();
    void addMapPoint(const cv::Scalar &color = cv::Scalar(200));
    void addMapPoint(const cv::Mat &desc, const cv::Scalar &color = cv::Scalar(200));

    std::shared_ptr<Frame> getKeyframe(const int kfid) const;
    std::shared_ptr<MapPoint> getMapPoint(const int lmid) const;

    void updateMapPoint(const int lmid, const Eigen::Vector3d &wpt, const double kfanch_invdepth=-1.);
    void addMapPointKfObs(const int lmid, const int kfid);

    bool setMapPointObs(const int lmid);

    void updateFrameCovisibility(Frame &frame);
    void mergeMapPoints(const int prevlmid, const int newlmid);

    void removeKeyframe(const int kfid);
    void removeMapPoint(const int lmid);
    void removeMapPointObs(const int lmid, const int kfid);
    void removeMapPointObs(MapPoint &lm, Frame &frame);

    void removeObsFromCurFrameById(const int lmid);
    void removeObsFromCurFrameByIdx(const int kpidx);

    void createKeyframe(const cv::Mat &im, const cv::Mat &imraw);

    void addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, Frame &frame);
    void addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, 
                const std::vector<int> &vscales, Frame &frame);
    void addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, 
                const std::vector<cv::Mat> &vdescs, Frame &frame);
    void addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, 
                const std::vector<int> &vscales, const std::vector<float> &vangles, 
                const std::vector<cv::Mat> &vdescs, Frame &frame);

    void extractKeypoints(const cv::Mat &im, const cv::Mat &imraw);

    void describeKeypoints(const cv::Mat &im, const std::vector<Keypoint> &vkps, 
                const std::vector<cv::Point2f> &vpts, 
                const std::vector<int> *pvscales = nullptr, 
                std::vector<float> *pvangles = nullptr);

    void kltStereoTracking(const std::vector<cv::Mat> &vleftpyr, 
                const std::vector<cv::Mat> &vrightpyr);

    void stereoMatching(Frame &frame, const std::vector<cv::Mat> &vleftpyr, 
                const std::vector<cv::Mat> &vrightpyr);

    void guidedMatching(Frame &frame);

    void triangulate(Frame &frame);
    void triangulateTemporal(Frame &frame);
    void triangulateStereo(Frame &frame);

    Eigen::Vector3d computeTriangulation(const Sophus::SE3d &T, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr);

    void getDepthDistribution(const Frame &frame, double &mean_depth, double &std_depth);

    void reset();
    
    int nlmid_, nkfid_;
    int nblms_, nbkfs_;

    std::shared_ptr<SlamParams> pslamstate_;
    std::shared_ptr<FeatureExtractor> pfeatextract_;
    std::shared_ptr<FeatureTracker> ptracker_;

    std::shared_ptr<Frame> pcurframe_;

    std::unordered_map<int, std::shared_ptr<Frame>> map_pkfs_;
    std::unordered_map<int, std::shared_ptr<MapPoint>> map_plms_;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud_;

    mutable std::mutex kf_mutex_, lm_mutex_;
    mutable std::mutex curframe_mutex_;

    mutable std::mutex map_mutex_, optim_mutex_;
};