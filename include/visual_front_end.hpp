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

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "slam_params.hpp"
#include "map_manager.hpp"
#include "feature_tracker.hpp"

class MotionModel {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void applyMotionModel(Sophus::SE3d &Twc, double time) {
        if( prev_time_ > 0 ) 
        {
            // Provided Twc and prevTwc should be equal here
            // as prevTwc is updated right after pose computation
            if( !(Twc * prevTwc_.inverse()).log().isZero(1.e-5) )
            {
                // Might happen in case of LC!
                // So update prevPose to stay consistent
                prevTwc_ = Twc;
            }

            double dt = (time - prev_time_);
            Twc = Twc * Sophus::SE3d::exp(log_relT_ * dt);
        }
    }

    void updateMotionModel(const Sophus::SE3d &Twc, double time) {
        if( prev_time_ < 0. ) {
            prev_time_ = time;
            prevTwc_ = Twc;
        } else {
            double dt = time - prev_time_;

            prev_time_ = time;
            
            if( dt < 0. ) {
                std::cerr << "\nGot image older than previous image! LEAVING!\n";
                exit(-1);
            }

            Sophus::SE3d Tprevcur = prevTwc_.inverse() * Twc;
            log_relT_ = Tprevcur.log() / dt;
            prevTwc_ = Twc;
        }
    }

    void reset() {
        prev_time_ = -1.;
        log_relT_ = Eigen::Matrix<double, 6, 1>::Zero();
    }
    

    double prev_time_ = -1.;

    Sophus::SE3d prevTwc_;
    Eigen::Matrix<double, 6, 1> log_relT_ = Eigen::Matrix<double, 6, 1>::Zero();
};

class VisualFrontEnd {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VisualFrontEnd() {}
    VisualFrontEnd(std::shared_ptr<SlamParams> pstate, std::shared_ptr<Frame> pframerame, 
        std::shared_ptr<MapManager> pmap, std::shared_ptr<FeatureTracker> ptracker);

    bool visualTracking(cv::Mat &iml, double time);

    bool trackMono(cv::Mat &im, double time);

    bool trackStereo(cv::Mat &iml, cv::Mat &imr, double time);

    void preprocessImage(cv::Mat &img_raw);

    void kltTracking();
    void kltTrackingFromKF();

    void epipolar2d2dFiltering();

    void computePose();

    float computeParallax(const int kfid, bool do_unrot=true, bool bmedian=true, bool b2donly=false);

    bool checkReadyForInit();
    bool checkNewKfReq();

    void createKeyframe();

    void applyMotion();
    void updateMotion();

    void resetFrame();
    void reset();

    std::shared_ptr<SlamParams> pslamstate_;
    std::shared_ptr<Frame> pcurframe_;
    std::shared_ptr<MapManager> pmap_;

    std::shared_ptr<FeatureTracker> ptracker_;

    cv::Mat left_raw_img_;
    cv::Mat cur_img_, prev_img_;
    std::vector<cv::Mat> cur_pyr_, prev_pyr_;
    std::vector<cv::Mat> kf_pyr_;
    
    MotionModel motion_model_;

    bool bp3preq_ = false; 
};
