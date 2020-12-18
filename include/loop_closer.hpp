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


#include <queue>
#include <deque>


#ifdef IBOW_LCD
#include <ibow_lcd/lcdetector.h>
#endif

#include "map_manager.hpp"
#include "optimizer.hpp"

class LoopCloser {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LoopCloser(std::shared_ptr<SlamParams> pslamstate, std::shared_ptr<MapManager> pmap);

    void run();
    void reset();

    void processLoopCandidate(int kfloopid);

    void knnMatching(const Frame &newkf, const Frame &lckf, std::vector<std::pair<int,int>> &vkplmids);
    
    bool epipolarFiltering(const Frame &newkf, const Frame &lckf, std::vector<std::pair<int,int>> &vkplmids, std::vector<int> &voutliers_idx);

    bool p3pRansac(const Frame &newkf, std::vector<std::pair<int,int>> &vkplmids, std::vector<int> &voutliers_idx, Sophus::SE3d &Twc);

    void trackLoopLocalMap(const Frame &newkf, const Frame &lckf, const Sophus::SE3d &Twc, const float maxdist, const float ratio, std::vector<std::pair<int,int>> &vkplmids);
    
    std::map<int,int> matchToMap(const Frame &frame, const Sophus::SE3d &Tcw, const float fmaxprojerr, const float fdistratio, 
                            const std::vector<int> &vmatchedkpids, std::unordered_set<int> &set_local_lmids);

    bool computePnP(const Frame &frame, const std::vector<std::pair<int,int>> &vkplmids, Sophus::SE3d &Twc, std::vector<int> &voutlier_idx);

    void removeOutliers(std::vector<std::pair<int,int>> &vkplmids, std::vector<int> &voutliers_idx);

    bool getNewKf();
    void processKeyframe();

    void addNewKf(const std::shared_ptr<Frame> &pkf, const cv::Mat &im);

#ifdef IBOW_LCD
    ibow_lcd::LCDetectorParams lcparams_; 
    ibow_lcd::LCDetector lcdetetector_;
#endif

    std::shared_ptr<SlamParams> pslamstate_;
    std::shared_ptr<MapManager> pmap_;

    std::unique_ptr<Optimizer> poptimizer_;

    std::shared_ptr<Frame> pnewkf_;
    cv::Mat newkfimg_;

    int kf_idx_ = -1;
    std::vector<int> vkfids_;

    bool bnewkfavailable_ = false;
    bool bexit_required_ = false;

    std::queue<std::pair<std::shared_ptr<Frame>, cv::Mat>> qpkfs_;
    std::mutex qkf_mutex_;
};