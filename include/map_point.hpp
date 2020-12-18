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


#include <unordered_map>
#include <set>
#include <mutex>

#include <Eigen/Core>

#include <opencv2/core.hpp>

class MapPoint {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MapPoint() {}
    MapPoint(const int lmid, const int kfid, const bool bobs=true);
    MapPoint(const int lmid, const int kfid, const cv::Mat &desc, const bool bobs=true);

    MapPoint(const int lmid, const int kfid, const cv::Scalar &color, const bool bobs=true);
    MapPoint(const int lmid, const int kfid, const cv::Mat &desc, const cv::Scalar &color, const bool bobs=true);

    void setPoint(const Eigen::Vector3d &ptxyz, const double kfanch_invdepth=-1.);
    Eigen::Vector3d getPoint() const;

    std::set<int> getKfObsSet() const;

    void addKfObs(const int kfid);
    void removeKfObs(const int kfid);

    void addDesc(const int kfid, const cv::Mat &d);

    bool isBad();

    float computeMinDescDist(const MapPoint &lm);

    // For using MapPoint in ordered containers
    bool operator< (const MapPoint &mp) const
    {
        return lmid_ < mp.lmid_;
    }

    // MapPoint id
    int lmid_;

    // True if seen in current frame
    bool isobs_;

    // True if MP has been init
    bool is3d_;

    // Set of observed KF ids
    std::set<int> set_kfids_;

    // 3D position
    Eigen::Vector3d ptxyz_;

    // Anchored position
    int kfid_;
    double invdepth_;

    // Mean desc and list of descs
    cv::Mat desc_;
    std::unordered_map<int, cv::Mat> map_kf_desc_;
    std::unordered_map<int, float> map_desc_dist_;

    // For vizu
    cv::Scalar color_ = cv::Scalar(200);

    mutable std::mutex pt_mutex;
};