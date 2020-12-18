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

#include <opencv2/core.hpp>

class MultiViewGeometry {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Triangulation methods
    // =====================

    // Generic
    static Eigen::Vector3d triangulate(const Sophus::SE3d &Tlr, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr);

    // OpenGV based
    static Eigen::Vector3d opengvTriangulate1(const Sophus::SE3d &Tlr, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr);
    static Eigen::Vector3d opengvTriangulate2(const Sophus::SE3d &Tlr, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr);

    // OpenCV based
    static Eigen::Vector3d opencvTriangulate(const Sophus::SE3d &Tlr, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr);
   
    // P3P - PnP methods
    // =================

    // Generic
    static bool p3pRansac(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs,
                        const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
                        const int nmaxiter, const float errth, const bool boptimize, const bool bdorandom,
                        const float fx, const float fy, Sophus::SE3d &Twc, std::vector<int> &voutliersidx, bool use_lmeds=false);

    // OpenGV based
    static bool opengvP3PRansac(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs,
                                const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
                                const int nmaxiter, const float errth, const bool boptimize, const bool bdorandom,
                                const float fx, const float fy, Sophus::SE3d &Twc, std::vector<int> &voutliersidx);

    static bool opengvP3PLMeds(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs,
                               const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
                               const int nmaxiter, const float errth, const bool boptimize, const bool bdorandom,
                               const float fx, const float fy, Sophus::SE3d &Twc, std::vector<int> &voutliersidx);

    // OpenCV based
    static bool opencvP3PRansac(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs,
                                const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
                                const int nmaxiter, const float errth, const float fx, const float fy, 
                                const bool boptimize, Sophus::SE3d &Twc, std::vector<int> &voutliersidx);

    // Ceres based
    static bool ceresPnP(const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > &vunkps, 
                        const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts, 
                        Sophus::SE3d &Twc, 
                        const int nmaxiter, const float chi2th, const bool buse_robust, const bool bapply_l2_after_robust, 
                        const float fx, const float fy, const float cx, const float cy, std::vector<int> &voutliersidx);

    static bool ceresPnP(const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > &vunkps, 
                        const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
                        const std::vector<int> &vscales, 
                        Sophus::SE3d &Twc,
                        const int nmaxiter, const float chi2th, const bool buse_robust, const bool bapply_l2_after_robust, 
                        const float fx, const float fy, const float cx, const float cy, std::vector<int> &voutliersidx);

    // 2D-2D Epipolar Geometry
    // =======================

    // Generic
    static bool compute5ptEssentialMatrix(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs1, 
                                        const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs2, 
                                        const int nmaxiter, const float errth, const bool boptimize, const bool bdorandom, 
                                        const float fx, const float fy, Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc, 
                                        std::vector<int> &voutliersidx);

    // OpenGV based
    static bool opengv5ptEssentialMatrix(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs1, 
                                        const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs2, 
                                        const int nmaxiter, const float errth, const bool boptimize, const bool bdorandom, 
                                        const float fx, const float fy, Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc, 
                                        std::vector<int> &voutliersidx);

    // OpenCV based
    static bool opencv5ptEssentialMatrix(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs1, 
                                        const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs2, 
                                        const int nmaxiter, const float errth, const bool boptimize, 
                                        const float fx, const float fy, Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc, 
                                        std::vector<int> &voutliersidx);
    // Misc.
    // =====
    
    static float computeSampsonDistance(const Eigen::Matrix3d &Frl, const Eigen::Vector3d &leftpt, const Eigen::Vector3d &rightpt);
    static float computeSampsonDistance(const Eigen::Matrix3d &Frl, const cv::Point2f &leftpt, const cv::Point2f &rightpt);

    static Eigen::Matrix3d computeFundamentalMat12(const Sophus::SE3d &Tw1, const Sophus::SE3d &Tw2, const Eigen::Matrix3d &K1, const Eigen::Matrix3d &K2);
    static Eigen::Matrix3d computeFundamentalMat12(const Sophus::SE3d &Tw1, const Sophus::SE3d &Tw2, const Eigen::Matrix3d &K);
};
