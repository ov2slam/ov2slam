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
#include <vector>
#include <unordered_set>

#include "map_manager.hpp"
#include "multi_view_geometry.hpp"
#include "optimizer.hpp"
#include "estimator.hpp"
#include "loop_closer.hpp"

struct Keyframe {
    int kfid_;
    cv::Mat imleft_, imright_;
    cv::Mat imleftraw_, imrightraw_;
    std::vector<cv::Mat> vpyr_imleft_, vpyr_imright_;
    bool is_stereo_;
    
    Keyframe()
        : kfid_(-1), is_stereo_(false)
    {}

    Keyframe(int kfid, const cv::Mat &imleftraw) 
        : kfid_(kfid), imleftraw_(imleftraw.clone()), is_stereo_(false)
    {}


    Keyframe(int kfid, const cv::Mat &imleftraw, const std::vector<cv::Mat> &vpyrleft, 
        const std::vector<cv::Mat> &vpyrright )
        : kfid_(kfid), imleftraw_(imleftraw.clone()) 
        , vpyr_imleft_(vpyrleft), vpyr_imright_(vpyrright)
        , is_stereo_(true)
    {}

    Keyframe(int kfid, const cv::Mat &imleftraw, const cv::Mat &imrightraw, 
        const std::vector<cv::Mat> &vpyrleft
         )
        : kfid_(kfid)
        , imleftraw_(imleftraw.clone())
        , imrightraw_(imrightraw.clone())
        , vpyr_imleft_(vpyrleft)
    {}

     void displayInfo() {
         std::cout << "\n\n Keyframe struct object !  Info : id #" << kfid_ << " - is stereo : " << is_stereo_;
         std::cout << " - imleft size : " << imleft_.size << " - imright size : " << imright_.size;
         std::cout << " - pyr left size : " << vpyr_imleft_.size() << " - pyr right size : " << vpyr_imright_.size() << "\n\n";
     }

     void releaseImages() {
         imleft_.release();
         imright_.release();
         imleftraw_.release();
         imrightraw_.release();
         vpyr_imleft_.clear();
         vpyr_imright_.clear();
     }
};

class Mapper {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Mapper() {}
    Mapper(std::shared_ptr<SlamParams> pslamstate, std::shared_ptr<MapManager> pmap, std::shared_ptr<Frame> pframe);

    void run();

    bool matchingToLocalMap(Frame &frame);
    std::map<int,int> matchToMap(const Frame &frame, const float fmaxprojerr, const float fdistratio, std::unordered_set<int> &set_local_lmids);
    void mergeMatches(const Frame &frame, const std::map<int,int> &map_kpids_lmids);
    bool p3pRansac(const Frame &frame, const std::map<int,int>& map_kpids_lmids, Sophus::SE3d &Twc, std::vector<int> &voutlier_ids);
    bool computePnP(const Frame &frame, const std::map<int,int>& map_kpids_lmids, Sophus::SE3d &Twc, std::vector<int> &voutlier_ids);

    void triangulate(Frame &frame);
    void triangulateTemporal(Frame &frame);
    void triangulateStereo(Frame &frame);

    bool triangulate(const Sophus::SE3d &T, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr, Eigen::Vector3d &wpt);

    Eigen::Vector3d computeTriangulation(const Sophus::SE3d &T, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr);

    void runFullBA();

    void runFullPoseGraph(std::vector<double*> &vtwc, std::vector<double*> &vqwc, std::vector<double*> &vtprevcur, std::vector<double*> &vqprevcur, std::vector<bool> &viskf);

    bool getNewKf(Keyframe &kf);
    void addNewKf(const Keyframe &kf);

    void reset();

    std::shared_ptr<SlamParams> pslamstate_;
    std::shared_ptr<MapManager> pmap_;
    std::shared_ptr<Frame> pcurframe_;

    std::shared_ptr<Estimator> pestimator_;
    std::shared_ptr<LoopCloser> ploopcloser_;

    bool bnewkfavailable_ = false;
    bool bwaiting_for_lc_ = false;
    bool bexit_required_ = false; 

    std::queue<Keyframe> qkfs_;

    std::mutex qkf_mutex_;
};