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


#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class FeatureTracker {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // FeatureTracker() {}
    FeatureTracker(int nmax_iter, float fmax_px_precision, cv::Ptr<cv::CLAHE> pclahe) 
        : klt_convg_crit_(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, nmax_iter, fmax_px_precision)
        , pclahe_(pclahe)
    {}

    // Forward-Backward KLT Tracking
    void fbKltTracking(const std::vector<cv::Mat> &vprevpyr, const std::vector<cv::Mat> &vcurpyr, int nwinsize, int nbpyrlvl, float ferr, float fmax_fbklt_dist,
        std::vector<cv::Point2f> &vpts, std::vector<cv::Point2f> &vpriorkps, std::vector<bool> &vkpstatus) const;
    
    void getLineMinSAD(const cv::Mat &iml, const cv::Mat &imr, const cv::Point2f &pt, const int nwinsize, float &xprior, float &l1err, bool bgoleft) const;

    bool inBorder(const cv::Point2f &pt, const cv::Mat &im) const;

    // KLT optim. parameter
    cv::TermCriteria klt_convg_crit_;

    cv::Ptr<cv::CLAHE> pclahe_;
};