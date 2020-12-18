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

#ifdef USE_OPENGV

#include <opengv/types.hpp>
#include <opengv/triangulation/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac/Lmeds.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#endif

#include "multi_view_geometry.hpp"
#include "ceres_parametrization.hpp"

#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>


// Triangulation methods
// =====================

// Generic
Eigen::Vector3d MultiViewGeometry::triangulate(const Sophus::SE3d &Tlr, 
    const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr)
{
    #ifdef USE_OPENGV
        return opengvTriangulate2(Tlr, bvl, bvr);
    #else
        return opencvTriangulate(Tlr, bvl, bvr);
    #endif
}

// OpenGV based

#ifdef USE_OPENGV
Eigen::Vector3d MultiViewGeometry::opengvTriangulate1(const Sophus::SE3d &Tlr, 
    const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr)
{
    opengv::bearingVectors_t bv1(1,bvl);
    opengv::bearingVectors_t bv2(1,bvr);
    opengv::rotation_t R12 = Tlr.rotationMatrix();
    opengv::translation_t t12 = Tlr.translation();

    opengv::relative_pose::CentralRelativeAdapter 
                    adapter(bv1, bv2, t12, R12);

    opengv::point_t pt = 
            opengv::triangulation::triangulate(adapter, 0);

    return pt;
}

Eigen::Vector3d MultiViewGeometry::opengvTriangulate2(const Sophus::SE3d &Tlr, 
    const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr)
{
    opengv::bearingVectors_t bv1(1,bvl);
    opengv::bearingVectors_t bv2(1,bvr);
    opengv::rotation_t R12 = Tlr.rotationMatrix();
    opengv::translation_t t12 = Tlr.translation();

    opengv::relative_pose::CentralRelativeAdapter 
                    adapter(bv1, bv2, t12, R12);

    opengv::point_t pt = 
            opengv::triangulation::triangulate2(adapter, 0);

    return pt;
}
#endif

// OpenCV based

Eigen::Vector3d MultiViewGeometry::opencvTriangulate(const Sophus::SE3d &Tlr, 
    const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr)
{
    std::vector<cv::Point2f> lpt, rpt;
    lpt.push_back( cv::Point2f(bvl.x()/bvl.z(), bvl.y()/bvl.z()) );
    rpt.push_back( cv::Point2f(bvr.x()/bvr.z(), bvr.y()/bvr.z()) );

    cv::Matx34f P0 = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);

    Sophus::SE3d Tcw = Tlr.inverse();
    Eigen::Matrix3d R = Tcw.rotationMatrix();
    Eigen::Vector3d t = Tcw.translation();

    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));

    cv::Mat campt;
    cv::triangulatePoints(P0, P1, lpt, rpt, campt);

    if( campt.col(0).at<float>(3) != 1. ) {
        campt.col(0) /= campt.col(0).at<float>(3);
    }

    Eigen::Vector3d pt(
                    campt.col(0).at<float>(0),
                    campt.col(0).at<float>(1),
                    campt.col(0).at<float>(2)
                    );

    return pt;
}


// Triangulation methods
// =====================

// Generic

bool MultiViewGeometry::p3pRansac(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
    const int nmaxiter, const float errth, const bool boptimize, const bool bdorandom,
    const float fx, const float fy, Sophus::SE3d &Twc, std::vector<int> &voutliersidx, 
    bool use_lmeds)
{
    #ifdef USE_OPENGV
        if( use_lmeds ) {
            return opengvP3PLMeds(bvs, vwpts, nmaxiter, errth, 
                    boptimize, bdorandom, fx, fy, Twc, voutliersidx);
        } else {
            return opengvP3PRansac(bvs, vwpts, nmaxiter, errth, 
                    boptimize, bdorandom, fx, fy, Twc, voutliersidx);
        }
    #else
        return opencvP3PRansac(bvs, vwpts, nmaxiter, errth, 
                    fx, fy, boptimize, Twc, voutliersidx);
    #endif
}

// OpenGV based

#ifdef USE_OPENGV
bool MultiViewGeometry::opengvP3PRansac(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
    const int nmaxiter, const float errth, const bool boptimize, const bool bdorandom,
    const float fx, const float fy, Sophus::SE3d &Twc, std::vector<int> &voutliersidx)
{
    assert( bvs.size() == vwpts.size() );

    size_t nb3dpts = bvs.size();

    if( nb3dpts < 4 ) {
        return false;
    }

    voutliersidx.reserve(nb3dpts);

    opengv::bearingVectors_t gvbvs;
    opengv::points_t gvwpt;
    gvbvs.reserve(nb3dpts);
    gvwpt.reserve(nb3dpts);

    for( size_t i = 0 ; i < nb3dpts ; i++ )
    {
        gvbvs.push_back(bvs.at(i));
        gvwpt.push_back(vwpts.at(i));
    }

    opengv::absolute_pose::CentralAbsoluteAdapter 
                            adapter(gvbvs, gvwpt);

    //Create an AbsolutePoseSac problem and Ransac
    //The method can be set to KNEIP, GAO or EPNP
    opengv::sac::Ransac<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;

    std::shared_ptr<
        opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> absposeproblem_ptr(
        new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
        adapter,
        opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::KNEIP,
        bdorandom));

    float focal = fx + fy;
    focal /= 2.;

    ransac.sac_model_ = absposeproblem_ptr;
    ransac.threshold_ = (1.0 - cos(atan(errth/focal)));
    ransac.max_iterations_ = nmaxiter;

    // Computing the pose from P3P
    ransac.computeModel(0);

    // If no solution found, return false
    if( ransac.inliers_.size() < 5 ) {
        return false;
    }

    // Might happen apparently...
    if( !Sophus::isOrthogonal(ransac.model_coefficients_.block<3,3>(0,0)) )
        return false;

    // Optimize the computed pose with inliers only
    opengv::transformation_t T_opt;

    if( boptimize ) {
        ransac.sac_model_->optimizeModelCoefficients(ransac.inliers_, ransac.model_coefficients_, T_opt);

        Twc.translation() = T_opt.rightCols(1);
        Twc.setRotationMatrix(T_opt.leftCols(3));
    } else {
        Twc.translation() = ransac.model_coefficients_.rightCols(1);
        Twc.setRotationMatrix(ransac.model_coefficients_.leftCols(3));
    }

    size_t k = 0;
    for( size_t i = 0 ; i < nb3dpts ; i++ ) {
        if( ransac.inliers_.at(k) == (int)i ) {
            k++;
            if( k == ransac.inliers_.size() ) {
                k = 0;
            }
        } else {
            voutliersidx.push_back(i);
        }
    }

    return true;
}


bool MultiViewGeometry::opengvP3PLMeds(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
    const int nmaxiter, const float errth, const bool boptimize, const bool bdorandom,
    const float fx, const float fy, Sophus::SE3d &Twc, std::vector<int> &voutliersidx)
{
    assert( bvs.size() == vwpts.size() );

    size_t nb3dpts = bvs.size();

    if( nb3dpts < 4 ) {
        return false;
    }

    voutliersidx.reserve(nb3dpts);

    opengv::bearingVectors_t gvbvs;
    opengv::points_t gvwpt;
    gvbvs.reserve(nb3dpts);
    gvwpt.reserve(nb3dpts);

    for( size_t i = 0 ; i < nb3dpts ; i++ )
    {
        gvbvs.push_back(bvs.at(i));
        gvwpt.push_back(vwpts.at(i));
    }
    
    opengv::absolute_pose::CentralAbsoluteAdapter 
                            adapter(gvbvs, gvwpt);

    //Create an AbsolutePoseSac problem and Ransac
    //The method can be set to KNEIP, GAO or EPNP
    opengv::sac::Lmeds<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;

    std::shared_ptr<
        opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> absposeproblem_ptr(
        new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
        adapter,
        opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::KNEIP,
        bdorandom));

    float focal = fx + fy;
    focal /= 2.;

    ransac.sac_model_ = absposeproblem_ptr;
    ransac.threshold_ = (1.0 - cos(atan(errth/focal)));
    ransac.max_iterations_ = nmaxiter;

    // Computing the pose from P3P
    ransac.computeModel(0);

    // If no solution found, return false
    if( ransac.inliers_.size() < 5 ) {
        return false;
    }

    // Might happen apparently...
    if( !Sophus::isOrthogonal(ransac.model_coefficients_.block<3,3>(0,0)) )
        return false;

    // Optimize the computed pose with inliers only
    opengv::transformation_t T_opt;

    if( boptimize ) {
        ransac.sac_model_->optimizeModelCoefficients(ransac.inliers_, ransac.model_coefficients_, T_opt);

        Twc.translation() = T_opt.rightCols(1);
        Twc.setRotationMatrix(T_opt.leftCols(3));
    } else {
        Twc.translation() = ransac.model_coefficients_.rightCols(1);
        Twc.setRotationMatrix(ransac.model_coefficients_.leftCols(3));
    }

    size_t k = 0;
    for( size_t i = 0 ; i < nb3dpts ; i++ ) {
        if( ransac.inliers_.at(k) == (int)i ) {
            k++;
            if( k == ransac.inliers_.size() ) {
                k = 0;
            }
        } else {
            voutliersidx.push_back(i);
        }
    }

    return true;
}
#endif

// OpenCV based

bool MultiViewGeometry::opencvP3PRansac(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
    const int nmaxiter, const float errth,  const float fx, const float fy, 
    const bool boptimize, Sophus::SE3d &Twc, std::vector<int> &voutliersidx)
{
    assert( bvs.size() == vwpts.size() );

    size_t nb3dpts = bvs.size();

    if( nb3dpts < 4 ) {
        return false;
    }

    voutliersidx.reserve(nb3dpts);

    std::vector<cv::Point2f> cvbvs;
    cvbvs.reserve(nb3dpts);

    std::vector<cv::Point3f> cvwpts;
    cvwpts.reserve(nb3dpts);

    for( size_t i=0 ; i < nb3dpts ; i++ )
    {
        cvbvs.push_back( 
                    cv::Point2f(
                        bvs.at(i).x()/bvs.at(i).z(), 
                        bvs.at(i).y()/bvs.at(i).z()
                        ) 
                    );

        cvwpts.push_back(
                    cv::Point3f(
                        vwpts.at(i).x(),
                        vwpts.at(i).y(),
                        vwpts.at(i).z()
                        )
                    );
    }

    // Using homoegeneous pts here so no dist or calib.
    cv::Mat D;
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);

    cv::Mat tvec, rvec;
    cv::Mat inliers;

    bool use_extrinsic_guess = false;
    float confidence = 0.99;

    float focal = (fx + fy) / 2.;

    cv::solvePnPRansac(
                cvwpts,
                cvbvs,
                K,
                D,
                rvec,
                tvec,
                use_extrinsic_guess,
                nmaxiter,
                errth / focal,
                confidence,
                inliers,
                cv::SOLVEPNP_P3P
                );

    if( inliers.rows == 0 ) {
        return false;
    }

    int k = 0;
    for( size_t i = 0 ; i < nb3dpts ; i++ ) {
        if( inliers.at<int>(k) == (int)i ) {
            k++;
            if( k == inliers.rows ) {
                k = 0;
            }
        } else {
            voutliersidx.push_back(i);
        }
    }

    if( voutliersidx.size() >= nb3dpts-5 ) {
        return false;
    }

    if( boptimize ) {
        use_extrinsic_guess = true;

        // Filter outliers
        std::vector<cv::Point2f> in_cvbvs;
        in_cvbvs.reserve(inliers.rows);

        std::vector<cv::Point3f> in_cvwpts;
        in_cvwpts.reserve(inliers.rows);

        for( int i=0 ; i < inliers.rows ; i++ )
        {
            in_cvbvs.push_back( cvbvs.at(inliers.at<int>(i)) );
            in_cvwpts.push_back( cvwpts.at(inliers.at<int>(i)) );
        }

        cv::solvePnP(
                in_cvwpts,
                in_cvbvs,
                K,
                D,
                rvec,
                tvec,
                use_extrinsic_guess,
                cv::SOLVEPNP_ITERATIVE
            );
    }

    // Store the resulting pose
    cv::Mat cvR;
    cv::Rodrigues(rvec, cvR);

    Eigen::Vector3d tcw;
    Eigen::Matrix3d Rcw;

    cv::cv2eigen(cvR, Rcw);
    cv::cv2eigen(tvec, tcw);

    Twc.translation() = -1. * Rcw.transpose() * tcw;
    Twc.setRotationMatrix( Rcw.transpose() );
    
    return true;
}

// Ceres Based

bool MultiViewGeometry::ceresPnP(
    const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > &vunkps,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
    Sophus::SE3d &Twc, const int nmaxiter, const float chi2th, const bool buse_robust, 
    const bool bapply_l2_after_robust, const float fx, const float fy, 
    const float cx, const float cy, std::vector<int> &voutliersidx)
{
    std::vector<int> vscales(vunkps.size(), 0);
    return ceresPnP(vunkps, vwpts, vscales, Twc, nmaxiter, chi2th, buse_robust, bapply_l2_after_robust, fx, fy, cx, cy, voutliersidx);
}

bool MultiViewGeometry::ceresPnP(
    const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > &vunkps,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
    const std::vector<int> &vscales, Sophus::SE3d &Twc, const int nmaxiter, 
    const float chi2th, const bool buse_robust, const bool bapply_l2_after_robust,
    const float fx, const float fy, const float cx, const float cy, 
    std::vector<int> &voutliersidx)
{
    assert( vunkps.size() == vwpts.size() );

    ceres::Problem problem;

    double chi2thrtsq = std::sqrt(chi2th);

    ceres::LossFunctionWrapper *loss_function;
    loss_function = new ceres::LossFunctionWrapper(new ceres::HuberLoss(chi2thrtsq), ceres::TAKE_OWNERSHIP);

    if( !buse_robust ) {
        loss_function->Reset(NULL, ceres::TAKE_OWNERSHIP);
    }

    size_t nbkps = vunkps.size();

    ceres::LocalParameterization *local_parameterization = new SE3LeftParameterization();

    PoseParametersBlock posepar = PoseParametersBlock(0, Twc);

    problem.AddParameterBlock(posepar.values(), 7, local_parameterization);

    std::vector<DirectLeftSE3::ReprojectionErrorSE3*> verrors_;
    std::vector<ceres::ResidualBlockId> vrids_;

    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        DirectLeftSE3::ReprojectionErrorSE3 *f = 
            new DirectLeftSE3::ReprojectionErrorSE3(
                    vunkps[i].x(), vunkps[i].y(),
                    fx, fy, cx, cy, vwpts.at(i),
                    std::pow(2.,vscales[i]));

        ceres::ResidualBlockId rid = problem.AddResidualBlock(f, loss_function, posepar.values());

        verrors_.push_back(f);
        vrids_.push_back(rid);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    options.num_threads = 1;
    options.max_num_iterations = nmaxiter;
    options.max_solver_time_in_seconds = 0.005;
    options.function_tolerance = 1.e-3;

    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // std::cout << summary.BriefReport() << std::endl;

    // std::cout << "\n Prev trans : " << twc.transpose();
    size_t nbbad = 0;

    for( size_t i = 0 ; i < nbkps ; i++ ) 
    {
        auto err = verrors_.at(i);
        if( err->chi2err_ > chi2th || !err->isdepthpositive_ ) 
        {
            if( bapply_l2_after_robust ) {
                auto rid = vrids_.at(i);
                problem.RemoveResidualBlock(rid);
            }
            voutliersidx.push_back(i);
            nbbad++;
        }
    }

    if( nbbad == nbkps ) {
        return false;
    }

    if( bapply_l2_after_robust && !voutliersidx.empty() ) {
        loss_function->Reset(NULL, ceres::TAKE_OWNERSHIP);
        ceres::Solve(options, &problem, &summary);
        // std::cout << summary.BriefReport() << std::endl;
    }

    Twc = posepar.getPose();
    
    return summary.IsSolutionUsable();
}


// 2D-2D Epipolar Geometry
// =======================

// Generic

bool MultiViewGeometry::compute5ptEssentialMatrix(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs1, 
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs2, 
    const int nmaxiter, const float errth, const bool boptimize, 
    const bool bdorandom, const float fx, const float fy, Eigen::Matrix3d &Rwc,
    Eigen::Vector3d &twc, std::vector<int> &voutliersidx)
{
    #ifdef USE_OPENGV
        return opengv5ptEssentialMatrix(bvs1, bvs2, nmaxiter, errth, boptimize, 
                                    bdorandom, fx, fy, Rwc, twc, voutliersidx);
    #else
        return opencv5ptEssentialMatrix(bvs1, bvs2, nmaxiter, errth, boptimize, 
                                    fx, fy, Rwc, twc, voutliersidx);
    #endif    
}

// OpenGV

#ifdef USE_OPENGV
bool MultiViewGeometry::opengv5ptEssentialMatrix(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs1, 
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs2, 
    const int nmaxiter, const float errth, const bool boptimize, 
    const bool bdorandom, const float fx, const float fy, Eigen::Matrix3d &Rwc, 
    Eigen::Vector3d &twc, std::vector<int> &voutliersidx)
{
    assert( bvs1.size() == bvs2.size() );

    size_t nbpts = bvs1.size();

    if( nbpts < 8 ) {
        return false;
    }

    voutliersidx.reserve(nbpts);

    opengv::bearingVectors_t vbv1, vbv2;
    vbv1.reserve(nbpts);
    vbv2.reserve(nbpts);

    for( size_t i = 0 ; i < nbpts ; i++ )
    {
        vbv1.push_back(bvs1.at(i));
        vbv2.push_back(bvs2.at(i));
    }
    
    //create a central relative adapter
    opengv::relative_pose::CentralRelativeAdapter 
                                adapter(vbv1, vbv2);

    opengv::sac::Ransac<
        opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem> ransac;

    std::shared_ptr<
        opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem> relposeproblem_ptr(
            new opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem(
                adapter,
                opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::NISTER,
                bdorandom));

    float focal = fx + fy;
    focal /= 2.;

    ransac.sac_model_ = relposeproblem_ptr;
    ransac.threshold_ = 2.0*(1.0 - cos(atan(errth/focal)));
    // ransac.threshold_ = (1.0 - cos(atan(errth/focal)));
    ransac.max_iterations_ = nmaxiter;

    ransac.computeModel(0);

    // If no solution found, return false
    if( ransac.inliers_.size() < 10 ) {
        return false;
    }

    twc = ransac.model_coefficients_.rightCols(1);
    Rwc = ransac.model_coefficients_.leftCols(3);

    // Optimize the computed pose with inliers only
    opengv::transformation_t T_opt;

    if( boptimize ) {
        ransac.sac_model_->optimizeModelCoefficients(ransac.inliers_, ransac.model_coefficients_, T_opt);

        Rwc = T_opt.leftCols(3);
        twc = T_opt.rightCols(1);
        twc.normalize();
    }

    size_t k = 0;
    for( size_t i = 0 ; i < nbpts ; i++ ) {
        if( ransac.inliers_.at(k) == (int)i ) {
            k++;
            if( k == ransac.inliers_.size() ) {
                k = 0;
            }
        } else {
            voutliersidx.push_back(i);
        }
    }

    return true;
}
#endif 

bool MultiViewGeometry::opencv5ptEssentialMatrix(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs1, 
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs2, 
    const int nmaxiter, const float errth, const bool boptimize, const float fx, const float fy,
    Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc, std::vector<int> &voutliersidx)
{
    assert( bvs1.size() == bvs2.size() );

    size_t nbpts = bvs1.size();

    if( nbpts < 5 ) {
        return false;
    }

    voutliersidx.reserve(nbpts);

    std::vector<cv::Point2f> cvbvs1, cvbvs2;
    cvbvs1.reserve(nbpts);
    cvbvs2.reserve(nbpts);

    for( size_t i = 0 ; i < nbpts ; i++ )
    {
        cvbvs1.push_back( 
                    cv::Point2f(
                        bvs1.at(i).x()/bvs1.at(i).z(), 
                        bvs1.at(i).y()/bvs1.at(i).z()
                        ) 
                    );

        cvbvs2.push_back( 
                    cv::Point2f(
                        bvs2.at(i).x()/bvs2.at(i).z(), 
                        bvs2.at(i).y()/bvs2.at(i).z()
                        ) 
                    );
    }

    // Using homoegeneous pts here so no dist or calib.
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);

    cv::Mat inliers;

    float confidence = 0.99;
    if( boptimize ) {
        confidence = 0.999;
    }

    float focal = (fx+fy) / 2.;

    cv::Mat E = 
        cv::findEssentialMat(
                    cvbvs1,
                    cvbvs2,
                    K,
                    cv::RANSAC,
                    confidence,
                    errth / focal,
                    inliers
                    );

    for( size_t i = 0 ; i < nbpts ; i++ ) {
        if( !inliers.at<uchar>(i) ) {
            voutliersidx.push_back(i);
        }
    }

    if( voutliersidx.size() >= nbpts-10 ) {
        return false;
    }

    cv::Mat tvec, cvR;

    cv::recoverPose(
                E,
                cvbvs1,
                cvbvs2,
                K,
                cvR,
                tvec,
                inliers
                );

    // Store the resulting pose
    Eigen::Vector3d tcw;
    Eigen::Matrix3d Rcw;

    cv::cv2eigen(cvR, Rcw);
    cv::cv2eigen(tvec, tcw);

    twc = -1. * Rcw.transpose() * tcw;
    Rwc = Rcw.transpose();

    return true;
}


// Misc.
// =====

float MultiViewGeometry::computeSampsonDistance(const Eigen::Matrix3d &Frl, const Eigen::Vector3d &leftpt, const Eigen::Vector3d &rightpt)
{
    float num = rightpt.transpose() * Frl * leftpt;
    num *= num;

    float x1, x2, y1, y2;
    x1 = (Frl.transpose() * rightpt).x();
    x2 = (Frl * leftpt).x();

    y1 = (Frl.transpose() * rightpt).y();
    y2 = (Frl * leftpt).y();
    
    float den = x1 * x1 + y1 * y1 + x2 * x2 + y2 * y2;

    return std::sqrt(num / den);
}


float MultiViewGeometry::computeSampsonDistance(const Eigen::Matrix3d &Frl, const cv::Point2f &leftpt, const cv::Point2f &rightpt)
{
    Eigen::Vector3d lpt(leftpt.x,leftpt.y,1.);
    Eigen::Vector3d rpt(rightpt.x,rightpt.y,1.);
    return computeSampsonDistance(Frl,lpt,rpt);
}


Eigen::Matrix3d MultiViewGeometry::computeFundamentalMat12(const Sophus::SE3d &Tw1, const Sophus::SE3d &Tw2,
                                                           const Eigen::Matrix3d &K1, const Eigen::Matrix3d &K2)
{
    // TODO Maxime Validate
    Sophus::SE3d T12 = Tw1.inverse() * Tw2;
    Eigen::Matrix3d skewt12 = Sophus::SO3d::hat(T12.translation());

    return K1.transpose().inverse() * skewt12 * T12.rotationMatrix() * K2.inverse();
}

Eigen::Matrix3d MultiViewGeometry::computeFundamentalMat12(const Sophus::SE3d &Tw1, const Sophus::SE3d &Tw2,
                                                           const Eigen::Matrix3d &K)
{
    return computeFundamentalMat12(Tw1, Tw2, K, K);
}