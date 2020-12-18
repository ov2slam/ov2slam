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

#include "optimizer.hpp"

#include "ceres_parametrization.hpp"

#include <thread>


void Optimizer::localBA(Frame &newframe, const bool buse_robust_cost)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.BA_SetupPb");

    // =================================
    //      Setup BA Problem
    // =================================

    ceres::Problem problem;
    ceres::LossFunctionWrapper *loss_function;
    
    // Chi2 thresh.
    const float mono_th = pslamstate_->robust_mono_th_;

    loss_function = new ceres::LossFunctionWrapper(new ceres::HuberLoss(std::sqrt(mono_th)), ceres::TAKE_OWNERSHIP);

    if( !buse_robust_cost ) {
        loss_function->Reset(nullptr, ceres::TAKE_OWNERSHIP);
    }

    // Thresh. score for optimizing / fixing
    // a KF in BA (cov score with new KF)
    int nmincovscore = pslamstate_->nmin_covscore_;

    // Do not optim is tracking is poor 
    // (hopefully will get better soon!)
    if( (int)newframe.nb3dkps_ < nmincovscore ) {
        return;
    }

    size_t nmincstkfs = 2;

    if( pslamstate_->stereo_ ) {
        nmincstkfs = 1;
    }

    size_t nbmono = 0;
    size_t nbstereo = 0;

    auto ordering = new ceres::ParameterBlockOrdering;

    std::unordered_map<int, PoseParametersBlock> map_id_posespar_;
    std::unordered_map<int, PointXYZParametersBlock> map_id_pointspar_;
    std::unordered_map<int, InvDepthParametersBlock> map_id_invptspar_;

    std::unordered_map<int, std::shared_ptr<MapPoint>> map_local_plms;
    std::unordered_map<int, std::shared_ptr<Frame>> map_local_pkfs;

    // Storing the factors and their residuals block ids 
    // for fast accessing when cheking for outliers
    std::vector<
        std::pair<ceres::CostFunction*, 
        std::pair<ceres::ResidualBlockId, 
            std::pair<int,int>
    >>> 
        vreprojerr_kfid_lmid, vright_reprojerr_kfid_lmid, vanchright_reprojerr_kfid_lmid;

    // Add the left cam calib parameters
    auto pcalibleft = newframe.pcalib_leftcam_;
    CalibParametersBlock calibpar(0, pcalibleft->fx_, pcalibleft->fy_, pcalibleft->cx_, pcalibleft->cy_);
    problem.AddParameterBlock(calibpar.values(), 4);
    ordering->AddElementToGroup(calibpar.values(), 1);

    problem.SetParameterBlockConstant(calibpar.values());

    // Prepare variables if STEREO mode
    auto pcalibright = newframe.pcalib_rightcam_;
    CalibParametersBlock rightcalibpar;
    
    Sophus::SE3d Trl, Tlr;
    PoseParametersBlock rlextrinpose(0, Trl);

    if( pslamstate_->stereo_ ) {
        // Right Intrinsic
        rightcalibpar = CalibParametersBlock(0, pcalibright->fx_, pcalibright->fy_, pcalibright->cx_, pcalibright->cy_);
        problem.AddParameterBlock(rightcalibpar.values(), 4);
        ordering->AddElementToGroup(rightcalibpar.values(), 1);

        problem.SetParameterBlockConstant(rightcalibpar.values());

        // Right Extrinsic
        Tlr = pcalibright->getExtrinsic();
        Trl = Tlr.inverse();
        rlextrinpose = PoseParametersBlock(0, Trl);

        ceres::LocalParameterization *local_param = new SE3LeftParameterization();

        problem.AddParameterBlock(rlextrinpose.values(), 7, local_param);
        ordering->AddElementToGroup(rlextrinpose.values(), 1);

        problem.SetParameterBlockConstant(rlextrinpose.values());
    }

    // Get the new KF covisible KFs
    std::map<int,int> map_covkfs = newframe.getCovisibleKfMap();

    // Add the new KF to the map with max score
    map_covkfs.emplace(newframe.kfid_, newframe.nb3dkps_);

    // Keep track of MPs no suited for BA for speed-up
    std::unordered_set<int> set_badlmids;

    std::unordered_set<int> set_lmids2opt;
    std::unordered_set<int> set_kfids2opt;
    std::unordered_set<int> set_cstkfids;

    if( pslamstate_->debug_ )
        std::cout << "\n >>> Local BA : new KF is #" << newframe.kfid_ 
            << " -- with covisible graph of size : " << map_covkfs.size();

    bool all_cst = false;

    // Go through the covisibility Kf map
    int nmaxkfid = map_covkfs.rbegin()->first;

    for( auto it = map_covkfs.rbegin() ; it != map_covkfs.rend() ; it++ ) {

        int kfid = it->first;
        int covscore = it->second;

        if( kfid > newframe.kfid_ ) {
            covscore = newframe.nbkps_;
        }

        auto pkf = pmap_->getKeyframe(kfid);

        if( pkf == nullptr ) {
            newframe.removeCovisibleKf(kfid);
            continue;
        }
        
        // Add every KF to BA problem
        map_id_posespar_.emplace(kfid, PoseParametersBlock(kfid, pkf->getTwc()));

        ceres::LocalParameterization *local_parameterization = new SE3LeftParameterization();

        problem.AddParameterBlock(map_id_posespar_.at(kfid).values(), 7, local_parameterization);
        ordering->AddElementToGroup(map_id_posespar_.at(kfid).values(), 1);

        // For those to optimize, get their 3D MPs
        // for the others, set them as constant
        if( covscore >= nmincovscore && !all_cst && kfid > 0 ) {
            set_kfids2opt.insert(kfid);
            for( const auto &kp : pkf->getKeypoints3d() ) {
                set_lmids2opt.insert(kp.lmid_);
            }
        } else {
            set_cstkfids.insert(kfid);
            problem.SetParameterBlockConstant(map_id_posespar_.at(kfid).values());
            all_cst = true;
        }

        map_local_pkfs.emplace(kfid, pkf);
    }



    // Go through the MPs to optimize
    for( const auto &lmid : set_lmids2opt ) {
        auto plm = pmap_->getMapPoint(lmid);

        if( plm == nullptr ) {
            continue;
        }

        if( plm->isBad() ) {
            set_badlmids.insert(lmid);
            continue;
        }

        map_local_plms.emplace(lmid, plm);

        if( !pslamstate_->buse_inv_depth_ )
        {
            map_id_pointspar_.emplace(lmid, PointXYZParametersBlock(lmid, plm->getPoint()));

            problem.AddParameterBlock(map_id_pointspar_.at(lmid).values(), 3);
            ordering->AddElementToGroup(map_id_pointspar_.at(lmid).values(), 0);
        }

        int kfanchid = -1;
        double unanch_u = -1.;
        double unanch_v = -1.;

        for( const auto &kfid : plm->getKfObsSet() ) {

            if( kfid > nmaxkfid ) {
                continue;
            }

            auto pkfit = map_local_pkfs.find(kfid);
            std::shared_ptr<Frame> pkf = nullptr;

            // Add the observing KF if not set yet
            if( pkfit == map_local_pkfs.end() ) 
            {
                pkf = pmap_->getKeyframe(kfid);
                if( pkf == nullptr ) {
                    pmap_->removeMapPointObs(kfid,plm->lmid_);
                    continue;
                }
                map_local_pkfs.emplace(kfid, pkf);
                map_id_posespar_.emplace(kfid, PoseParametersBlock(kfid, pkf->getTwc()));

                ceres::LocalParameterization *local_parameterization = new SE3LeftParameterization();

                problem.AddParameterBlock(map_id_posespar_.at(kfid).values(), 7, local_parameterization);
                ordering->AddElementToGroup(map_id_posespar_.at(kfid).values(), 1);

                set_cstkfids.insert(kfid);
                problem.SetParameterBlockConstant(map_id_posespar_.at(kfid).values());

            } else {
                pkf = pkfit->second;
            }

            auto kp = pkf->getKeypointById(lmid);

            if( kp.lmid_ != lmid ) {
                pmap_->removeMapPointObs(lmid, kfid);
                continue;
            }

            if( pslamstate_->buse_inv_depth_ ) {
                if( kfanchid < 0 ) {
                    kfanchid = kfid;
                    unanch_u = kp.unpx_.x;
                    unanch_v = kp.unpx_.y;
                    double zanch = (pkf->getTcw() * plm->getPoint()).z();
                    map_id_invptspar_.emplace(lmid, InvDepthParametersBlock(lmid, kfanchid, zanch));

                    problem.AddParameterBlock(map_id_invptspar_.at(lmid).values(), 1);
                    ordering->AddElementToGroup(map_id_invptspar_.at(lmid).values(), 0);

                    if( kp.is_stereo_ ) 
                    {
                        ceres::CostFunction *f = new DirectLeftSE3::ReprojectionErrorRightAnchCamKSE3AnchInvDepth(
                                kp.runpx_.x, kp.runpx_.y, unanch_u, unanch_v, 
                                std::pow(2.,kp.scale_)
                            );

                        ceres::ResidualBlockId rid = problem.AddResidualBlock(
                                f, loss_function, 
                                calibpar.values(), rightcalibpar.values(),
                                rlextrinpose.values(), 
                                map_id_invptspar_.at(lmid).values()
                            );

                        vanchright_reprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,lmid))));
                        nbstereo++;
                    } else {
                        nbmono++;
                    }
                    continue;
                }
            }

            ceres::CostFunction *f;
            ceres::ResidualBlockId rid;

            // Add a visual factor between KF-MP nodes
            if( kp.is_stereo_ ) {
                if( pslamstate_->buse_inv_depth_ ) {

                    f = new DirectLeftSE3::ReprojectionErrorKSE3AnchInvDepth(
                                kp.unpx_.x, kp.unpx_.y, unanch_u, unanch_v, 
                                std::pow(2.,kp.scale_)
                            );

                    rid = problem.AddResidualBlock(
                                f, loss_function, 
                                calibpar.values(),
                                map_id_posespar_.at(kfanchid).values(),
                                map_id_posespar_.at(kfid).values(), 
                                map_id_invptspar_.at(lmid).values()
                            );
                        
                    vreprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid, lmid))));

                    f = new DirectLeftSE3::ReprojectionErrorRightCamKSE3AnchInvDepth(
                            kp.runpx_.x, kp.runpx_.y, unanch_u, unanch_v, 
                            std::pow(2.,kp.scale_)
                        );

                    rid = problem.AddResidualBlock(
                            f, loss_function, 
                            calibpar.values(), rightcalibpar.values(),
                            map_id_posespar_.at(kfanchid).values(), 
                            map_id_posespar_.at(kfid).values(), 
                            rlextrinpose.values(), 
                            map_id_invptspar_.at(lmid).values()
                        );
                    
                    vright_reprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,lmid))));
                    nbstereo++;
                    continue;

                } else {
                    f = new DirectLeftSE3::ReprojectionErrorKSE3XYZ(
                                kp.unpx_.x, kp.unpx_.y, std::pow(2.,kp.scale_)
                            );

                    rid = problem.AddResidualBlock(
                                f, loss_function, 
                                calibpar.values(),
                                map_id_posespar_.at(kfid).values(), 
                                map_id_pointspar_.at(lmid).values()
                            );
                        
                    vreprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,lmid))));

                    f = new DirectLeftSE3::ReprojectionErrorRightCamKSE3XYZ(
                            kp.runpx_.x, kp.runpx_.y, std::pow(2.,kp.scale_)
                        );

                    rid = problem.AddResidualBlock(
                            f, loss_function, 
                            rightcalibpar.values(),
                            map_id_posespar_.at(kfid).values(), 
                            rlextrinpose.values(), 
                            map_id_pointspar_.at(lmid).values()
                        );
                    
                    vright_reprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,lmid))));
                    nbstereo++;
                    continue;
                }
            } 
            else {
                if( pslamstate_->buse_inv_depth_ ) {
                    f = new DirectLeftSE3::ReprojectionErrorKSE3AnchInvDepth(
                                kp.unpx_.x, kp.unpx_.y, unanch_u, unanch_v, 
                                std::pow(2.,kp.scale_)
                            );

                    rid = problem.AddResidualBlock(
                                f, loss_function, 
                                calibpar.values(),
                                map_id_posespar_.at(kfanchid).values(),
                                map_id_posespar_.at(kfid).values(), 
                                map_id_invptspar_.at(lmid).values()
                            );
                } else {
                    f = new DirectLeftSE3::ReprojectionErrorKSE3XYZ(
                                kp.unpx_.x, kp.unpx_.y, std::pow(2.,kp.scale_)
                            );

                    rid = problem.AddResidualBlock(
                                f, loss_function, calibpar.values(), 
                                map_id_posespar_.at(kfid).values(), 
                                map_id_pointspar_.at(lmid).values()
                            );
                }

                vreprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,kp.lmid_))));

                nbmono++;
            }
        }
    }

    // Ensure the gauge is fixed
    size_t nbcstkfs = set_cstkfids.size();

    // At least two fixed KF in mono / one fixed KF in Stereo / RGB-D
    if( nbcstkfs < nmincstkfs ) {
        for( auto it = map_local_pkfs.begin() ; nbcstkfs < nmincstkfs && it != map_local_pkfs.end() ; it++ ) 
        {
            problem.SetParameterBlockConstant(map_id_posespar_.at(it->first).values());
            set_cstkfids.insert(it->first);
            nbcstkfs++;
        }
    }
    
    size_t nbkfstot = map_local_pkfs.size();
    size_t nbkfs2opt = nbkfstot - nbcstkfs;
    size_t nblms2opt = map_local_plms.size();

    if( pslamstate_->debug_ ) {
        std::cout << "\n\n >>> LocalBA problem setup!";
        std::cout << "\n >>> Kfs added (opt / tot) : " << nbkfs2opt 
            << " / " << nbkfstot;
        std::cout << "\n >>> MPs added : " << nblms2opt;
        std::cout << "\n >>> Measuremetns added (mono / stereo) : " 
            << nbmono << " / " << nbstereo;

        std::cout << "\n\n >>> KFs added : ";
        for( const auto &id_pkf : map_local_pkfs ) {
            std::cout << " KF #" << id_pkf.first << " (cst : " << set_cstkfids.count(id_pkf.first) << "), ";
        }

        std::cout << "\n\n >>> KFs in cov map : ";
        for( const auto &id_score : map_covkfs ) {
            std::cout << " KF #" << id_score.first << " (cov score : " << id_score.second << "), ";
        }
    }

    // =================================
    //      Solve BA Problem
    // =================================

    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);

    if( pslamstate_->use_sparse_schur_ ) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
    } else {
        options.linear_solver_type = ceres::DENSE_SCHUR;
    }

    if( pslamstate_->use_dogleg_ ) {
        options.trust_region_strategy_type = ceres::DOGLEG;
        if( pslamstate_->use_subspace_dogleg_ ) {
            options.dogleg_type = ceres::DoglegType::SUBSPACE_DOGLEG;
        } else {
            options.dogleg_type = ceres::DoglegType::TRADITIONAL_DOGLEG;
        }
    } else {
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    }

    if( pslamstate_->use_nonmonotic_step_ ) {
        options.use_nonmonotonic_steps = true;
    }

    options.num_threads = 1;

    options.max_num_iterations = 5;
    options.function_tolerance = 1.e-3;
    options.max_solver_time_in_seconds = 0.2;

    if( !pslamstate_->bforce_realtime_ ) {
        options.max_solver_time_in_seconds *= 2.;
    }
    
    options.minimizer_progress_to_stdout = pslamstate_->debug_;

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.BA_SetupPb");
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.BA_Optimize");

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if( pslamstate_->debug_ )
        std::cout << summary.FullReport() << std::endl;

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.BA_Optimize");


    // =================================
    //      Remove outliers
    // =================================

    size_t nbbadobsmono = 0;
    size_t nbbadobsrightcam = 0;

    std::vector<std::pair<int,int>> vbadkflmids;
    std::vector<std::pair<int,int>> vbadstereokflmids;
    vbadkflmids.reserve(vreprojerr_kfid_lmid.size() / 10);
    vbadstereokflmids.reserve(vright_reprojerr_kfid_lmid.size() / 10);

    for( auto it = vreprojerr_kfid_lmid.begin() 
        ; it != vreprojerr_kfid_lmid.end(); )
    {
        bool bbigchi2 = true;
        bool bdepthpos = true;
        if( pslamstate_->buse_inv_depth_ ) {
            auto *err = static_cast<DirectLeftSE3::ReprojectionErrorKSE3AnchInvDepth*>(it->first);
            bbigchi2 = err->chi2err_ > mono_th;
            bdepthpos = err->isdepthpositive_;
        }
        else {
            auto *err = static_cast<DirectLeftSE3::ReprojectionErrorKSE3XYZ*>(it->first);
            bbigchi2 = err->chi2err_ > mono_th;
            bdepthpos = err->isdepthpositive_;
        }
        
        if( bbigchi2 || !bdepthpos )
        {
            if( pslamstate_->apply_l2_after_robust_ ) {
                auto rid = it->second.first;
                problem.RemoveResidualBlock(rid);
            }
            int lmid = it->second.second.second;
            int kfid = it->second.second.first;
            vbadkflmids.push_back(std::pair<int,int>(kfid,lmid));
            set_badlmids.insert(lmid);
            nbbadobsmono++;

            it = vreprojerr_kfid_lmid.erase(it);
        } else {
            it++;
        }
    }

    for( auto it = vright_reprojerr_kfid_lmid.begin() 
    ; it != vright_reprojerr_kfid_lmid.end() ; )
    {
        bool bbigchi2 = true;
        bool bdepthpos = true;
        if( pslamstate_->buse_inv_depth_ ) {
            auto *err = static_cast<DirectLeftSE3::ReprojectionErrorRightCamKSE3AnchInvDepth*>(it->first);
            bbigchi2 = err->chi2err_ > mono_th;
            bdepthpos = err->isdepthpositive_;
        }
        else {
            auto *err = static_cast<DirectLeftSE3::ReprojectionErrorRightCamKSE3XYZ*>(it->first);
            bbigchi2 = err->chi2err_ > mono_th;
            bdepthpos = err->isdepthpositive_;
        }
        if( bbigchi2 || !bdepthpos )
        {
            if( pslamstate_->apply_l2_after_robust_ ) {
                auto rid = it->second.first;
                problem.RemoveResidualBlock(rid);
            }
            int lmid = it->second.second.second;
            int kfid = it->second.second.first;
            vbadstereokflmids.push_back(std::pair<int,int>(kfid,lmid));
            set_badlmids.insert(lmid);
            nbbadobsrightcam++;

            it = vright_reprojerr_kfid_lmid.erase(it);
        } else {
            it++;
        }
    }

    for( auto it = vanchright_reprojerr_kfid_lmid.begin() 
    ; it != vanchright_reprojerr_kfid_lmid.end() ; )
    {
        bool bbigchi2 = true;
        bool bdepthpos = true;
        auto *err = static_cast<DirectLeftSE3::ReprojectionErrorRightAnchCamKSE3AnchInvDepth*>(it->first);
        bbigchi2 = err->chi2err_ > mono_th;
        bdepthpos = err->isdepthpositive_;

        if( bbigchi2 || !bdepthpos )
        {
            if( pslamstate_->apply_l2_after_robust_ ) {
                auto rid = it->second.first;
                problem.RemoveResidualBlock(rid);
            }
            int lmid = it->second.second.second;
            int kfid = it->second.second.first;
            vbadstereokflmids.push_back(std::pair<int,int>(kfid,lmid));
            set_badlmids.insert(lmid);
            nbbadobsrightcam++;

            it = vanchright_reprojerr_kfid_lmid.erase(it);
        } else {
            it++;
        }
    }

    size_t nbbadobs = nbbadobsmono + nbbadobsrightcam;

    // =================================
    //      Refine BA Solution
    // =================================

    bool bl2optimdone = false;

    // Refine without Robust cost if req.
    if( pslamstate_->apply_l2_after_robust_ && buse_robust_cost 
        && !stopLocalBA() && nbbadobs > 0 )
    {
        if( !vreprojerr_kfid_lmid.empty() && !vright_reprojerr_kfid_lmid.empty() ) {
            loss_function->Reset(nullptr, ceres::TAKE_OWNERSHIP);
        }

        options.max_num_iterations = 10;
        options.function_tolerance = 1.e-3;
        options.max_solver_time_in_seconds /= 2.;
        // options.max_solver_time_in_seconds = 0.05;
        
        if( pslamstate_->debug_ )
            Profiler::Start("2.BA_L2-Refinement");

        ceres::Solve(options, &problem, &summary);

        bl2optimdone = true;

        if( pslamstate_->debug_ )
            std::cout << summary.FullReport() << std::endl;

        if( pslamstate_->debug_ )
            Profiler::StopAndDisplay(pslamstate_->debug_, "2.BA_L2-Refinement");
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.BA_Update");

    // =================================
    //      Remove outliers
    // =================================

    // Remove Bad Observations
    if( bl2optimdone )
    {
        // std::lock_guard<std::mutex> lock(pmap_->map_mutex_);

        for( auto it = vreprojerr_kfid_lmid.begin() 
            ; it != vreprojerr_kfid_lmid.end(); )
        {
            bool bbigchi2 = true;
            bool bdepthpos = true;
            if( pslamstate_->buse_inv_depth_ ) {
                auto *err = static_cast<DirectLeftSE3::ReprojectionErrorKSE3AnchInvDepth*>(it->first);
                bbigchi2 = err->chi2err_ > mono_th;
                bdepthpos = err->isdepthpositive_;
            }
            else {
                auto *err = static_cast<DirectLeftSE3::ReprojectionErrorKSE3XYZ*>(it->first);
                bbigchi2 = err->chi2err_ > mono_th;
                bdepthpos = err->isdepthpositive_;
            }
            
            if( bbigchi2 || !bdepthpos )
            {
                int lmid = it->second.second.second;
                int kfid = it->second.second.first;
                vbadkflmids.push_back(std::pair<int,int>(kfid,lmid));
                set_badlmids.insert(lmid);
                nbbadobsmono++;

                it = vreprojerr_kfid_lmid.erase(it);
            } else {
                it++;
            }
        }

        if( !vright_reprojerr_kfid_lmid.empty() ) {
            for( auto it = vright_reprojerr_kfid_lmid.begin() 
            ; it != vright_reprojerr_kfid_lmid.end(); )
            {
                bool bbigchi2 = true;
                bool bdepthpos = true;
                if( pslamstate_->buse_inv_depth_ ) {
                    auto *err = static_cast<DirectLeftSE3::ReprojectionErrorRightCamKSE3AnchInvDepth*>(it->first);
                    bbigchi2 = err->chi2err_ > mono_th;
                    bdepthpos = err->isdepthpositive_;
                }
                else {
                    auto *err = static_cast<DirectLeftSE3::ReprojectionErrorRightCamKSE3XYZ*>(it->first);
                    bbigchi2 = err->chi2err_ > mono_th;
                    bdepthpos = err->isdepthpositive_;
                }
                if( bbigchi2 || !bdepthpos )
                {
                    if( pslamstate_->apply_l2_after_robust_ ) {
                        auto rid = it->second.first;
                        problem.RemoveResidualBlock(rid);
                    }
                    int lmid = it->second.second.second;
                    int kfid = it->second.second.first;
                    vbadstereokflmids.push_back(std::pair<int,int>(kfid,lmid));
                    set_badlmids.insert(lmid);
                    nbbadobsrightcam++;

                    it = vright_reprojerr_kfid_lmid.erase(it);
                } else {
                    it++;
                }
            }
        }


        if( !vanchright_reprojerr_kfid_lmid.empty() ) {
            for( auto it = vanchright_reprojerr_kfid_lmid.begin() 
            ; it != vanchright_reprojerr_kfid_lmid.end(); )
            {
                bool bbigchi2 = true;
                bool bdepthpos = true;
                auto *err = static_cast<DirectLeftSE3::ReprojectionErrorRightAnchCamKSE3AnchInvDepth*>(it->first);
                bbigchi2 = err->chi2err_ > mono_th;
                bdepthpos = err->isdepthpositive_;

                if( bbigchi2 || !bdepthpos )
                {
                    if( pslamstate_->apply_l2_after_robust_ ) {
                        auto rid = it->second.first;
                        problem.RemoveResidualBlock(rid);
                    }
                    int lmid = it->second.second.second;
                    int kfid = it->second.second.first;
                    vbadstereokflmids.push_back(std::pair<int,int>(kfid,lmid));
                    set_badlmids.insert(lmid);
                    nbbadobsrightcam++;

                    it = vanchright_reprojerr_kfid_lmid.erase(it);
                } else {
                    it++;
                }
            }
        }
    }

    // =================================
    //      Update State Parameters
    // =================================

    std::lock_guard<std::mutex> lock(pmap_->map_mutex_);

    for( const auto &badkflmid : vbadstereokflmids ) {
        int kfid = badkflmid.first;
        int lmid = badkflmid.second;
        auto it = map_local_pkfs.find(kfid);
        if( it != map_local_pkfs.end() ) {
            it->second->removeStereoKeypointById(lmid);
        }
        set_badlmids.insert(lmid);
    }

    for( const auto &badkflmid : vbadkflmids ) {
        int kfid = badkflmid.first;
        int lmid = badkflmid.second;
        auto it = map_local_pkfs.find(kfid);
        if( it != map_local_pkfs.end() ) {
            pmap_->removeMapPointObs(lmid,kfid);
        }
        if( kfid == pmap_->pcurframe_->kfid_ ) {
            pmap_->removeObsFromCurFrameById(lmid);
        }
        set_badlmids.insert(lmid);
    }

    // Update KFs
    for( const auto &kfid_pkf : map_local_pkfs )
    {
        int kfid = kfid_pkf.first;

        if( set_cstkfids.count(kfid) ) {
            continue;
        }
        
        auto pkf = kfid_pkf.second;

        if( pkf == nullptr ) {
            continue;
        }

        // auto optkfpose = map_id_posespar_.at(kfid);
        auto it = map_id_posespar_.find(kfid);
        if( it != map_id_posespar_.end() ) {
            pkf->setTwc(it->second.getPose());
        }
    }

    // Update MPs
    for( const auto &lmid_plm : map_local_plms )
    {
        int lmid = lmid_plm.first;
        auto plm = lmid_plm.second;

        if( plm == nullptr ) {
            set_badlmids.erase(lmid);
            continue;
        }

        if( plm->isBad() ) {
            pmap_->removeMapPoint(lmid);
            set_badlmids.erase(lmid);
            continue;
        } 

        // Map Point Culling
        auto kfids = plm->getKfObsSet();
        if( kfids.size() < 3 ) {
            if( plm->kfid_ < newframe.kfid_-3 && !plm->isobs_ ) {
                pmap_->removeMapPoint(lmid);
                set_badlmids.erase(lmid);
                continue;
            }
        }

        if( pslamstate_->buse_inv_depth_ ) 
        {
            auto invptit = map_id_invptspar_.find(lmid);
            if( invptit == map_id_invptspar_.end() ) {
                set_badlmids.insert(lmid);
                continue;
            }
            double zanch = 1. / invptit->second.getInvDepth();
            if( zanch <= 0. ) {
                pmap_->removeMapPoint(lmid);
                set_badlmids.erase(lmid);
                continue;
            }

            auto it = map_local_pkfs.find(plm->kfid_);
            if( it == map_local_pkfs.end() ) {
                set_badlmids.insert(lmid);
                continue;
            }
            auto pkfanch = it->second;
            
            if( pkfanch != nullptr ) {
                auto kp = pkfanch->getKeypointById(lmid);
                Eigen::Vector3d uvpt(kp.unpx_.x,kp.unpx_.y,1.); 
                Eigen::Vector3d optwpt = pkfanch->getTwc() * (zanch * pkfanch->pcalib_leftcam_->iK_ * uvpt);
                pmap_->updateMapPoint(lmid, optwpt, invptit->second.getInvDepth());
            } else {
                set_badlmids.insert(lmid);
            }
        } 
        else {
            auto optlmit = map_id_pointspar_.find(lmid);
            if( optlmit != map_id_pointspar_.end() ) {
                pmap_->updateMapPoint(lmid, optlmit->second.getPoint());
            } else {
                set_badlmids.insert(lmid);
            }
        }
    }

    // Map Point Culling for bad Obs.
    size_t nbbadlm = 0;
    for( const auto &lmid : set_badlmids ) {

        std::shared_ptr<MapPoint> plm;
        auto plmit = map_local_plms.find(lmid);
        if( plmit == map_local_plms.end() ) {
            plm = pmap_->getMapPoint(lmid);
        } else {
            plm = plmit->second;
        }
        if( plm == nullptr ) {
            continue;
        }
        
        if( plm->isBad() ) {
            pmap_->removeMapPoint(lmid);
            nbbadlm++;
        } else {
            auto set_cokfs = plm->getKfObsSet();
            if( set_cokfs.size() < 3 ) {
                if( plm->kfid_ < newframe.kfid_-3 && !plm->isobs_ ) {
                    pmap_->removeMapPoint(lmid);
                    nbbadlm++;
                }
            }
        }
    }

    nbbadobs = nbbadobsmono + nbbadobsrightcam;

    if( pslamstate_->debug_ ) {
        std::cout << "\n \t>>> localBA() --> Nb of bad obs / nb removed MP : " 
            << nbbadobs << " / " << nbbadlm;
        std::cout << "\n \t>>> localBA() --> Nb of bad obs mono / right : " 
            << nbbadobsmono << " / " << nbbadobsrightcam;
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.BA_Update");

    bstop_localba_ = false;
}


void Optimizer::looseBA(int inikfid, const int nkfid, const bool buse_robust_cost)
{
    // =================================
    //      Setup BA Problem
    // =================================

    Frame &newframe = *pmap_->getKeyframe(nkfid);

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.LC_LooseBA_setup");

    ceres::Problem problem;
    ceres::LossFunctionWrapper *loss_function;
    
    // Chi2 thresh.
    const float mono_th = pslamstate_->robust_mono_th_;

    loss_function = new ceres::LossFunctionWrapper(new ceres::HuberLoss(std::sqrt(mono_th)), ceres::TAKE_OWNERSHIP);

    if( !buse_robust_cost ) {
        loss_function->Reset(nullptr, ceres::TAKE_OWNERSHIP);
    }

    size_t nmincstkfs = 2;

    if( pslamstate_->stereo_ ) {
        nmincstkfs = 1;
    }

    size_t nbmono = 0;
    size_t nbstereo = 0;

    auto ordering = new ceres::ParameterBlockOrdering;

    std::unordered_map<int, PoseParametersBlock> map_id_posespar_;
    std::unordered_map<int, PointXYZParametersBlock> map_id_pointspar_;
    std::unordered_map<int, InvDepthParametersBlock> map_id_invptspar_;

    std::unordered_map<int, std::shared_ptr<MapPoint>> map_local_plms;
    std::unordered_map<int, std::shared_ptr<Frame>> map_local_pkfs;

    // Storing the factors and their residuals block ids 
    // for fast accessing when cheking for outliers
    std::vector<
        std::pair<ceres::CostFunction*, 
        std::pair<ceres::ResidualBlockId, 
            std::pair<int,int>
    >>> 
        vreprojerr_kfid_lmid, vright_reprojerr_kfid_lmid, vanchright_reprojerr_kfid_lmid;

    // Add the left cam calib parameters
    auto pcalibleft = newframe.pcalib_leftcam_;
    CalibParametersBlock calibpar(0, pcalibleft->fx_, pcalibleft->fy_, pcalibleft->cx_, pcalibleft->cy_);
    problem.AddParameterBlock(calibpar.values(), 4);
    ordering->AddElementToGroup(calibpar.values(), 1);

    problem.SetParameterBlockConstant(calibpar.values());
    
    // Prepare variables if STEREO mode
    auto pcalibright = newframe.pcalib_rightcam_;
    CalibParametersBlock rightcalibpar;
    
    Sophus::SE3d Trl, Tlr;
    PoseParametersBlock rlextrinpose(0, Trl);

    if( pslamstate_->stereo_ ) {
        // Right Intrinsic
        rightcalibpar = CalibParametersBlock(0, pcalibright->fx_, pcalibright->fy_, pcalibright->cx_, pcalibright->cy_);
        problem.AddParameterBlock(rightcalibpar.values(), 4);
        ordering->AddElementToGroup(rightcalibpar.values(), 1);

        problem.SetParameterBlockConstant(rightcalibpar.values());

        // Right Extrinsic
        Tlr = pcalibright->getExtrinsic();
        Trl = Tlr.inverse();
        rlextrinpose = PoseParametersBlock(0, Trl);

        ceres::LocalParameterization *local_param = new SE3LeftParameterization();

        problem.AddParameterBlock(rlextrinpose.values(), 7, local_param);
        ordering->AddElementToGroup(rlextrinpose.values(), 1);

        problem.SetParameterBlockConstant(rlextrinpose.values());
    }

    // Keep track of MPs no suited for BA for speed-up
    std::unordered_set<int> set_badlmids;

    std::set<int> set_kfids2opt;
    std::set<int> set_lmids2opt;
    std::set<int> set_cstkfids;

    // Add the KFs to optimize to BA (new KF + cov KFs)
    // for( const auto &kfid_pkf : pmap_->map_pkfs_ )
    for( int kfid = inikfid ; kfid <= nkfid ; kfid++ )
    {
        // Add every KF to BA problem
        auto pkf = pmap_->getKeyframe(kfid);

        if( pkf == nullptr ) {
            continue;
        }

        map_id_posespar_.emplace(pkf->kfid_, PoseParametersBlock(pkf->kfid_, pkf->getTwc()));

        ceres::LocalParameterization *local_parameterization = new SE3LeftParameterization();

        problem.AddParameterBlock(map_id_posespar_.at(pkf->kfid_).values(), 7, local_parameterization);
        ordering->AddElementToGroup(map_id_posespar_.at(pkf->kfid_).values(), 1);

        if( set_cstkfids.size() < nmincstkfs ) {
            set_cstkfids.insert(pkf->kfid_);
            problem.SetParameterBlockConstant(map_id_posespar_.at(pkf->kfid_).values());
        } else {
            // Add observed MPs for KFs' to optimize
            set_kfids2opt.insert(pkf->kfid_);
            for( const auto &kp : pkf->getKeypoints3d() ) {
                set_lmids2opt.insert(kp.lmid_);
            }
        }

        map_local_pkfs.emplace(pkf->kfid_, pkf);
    }


    // Go through the MPs to optimize
    for( const auto &lmid : set_lmids2opt ) {
        auto plm = pmap_->getMapPoint(lmid);

        if( plm == nullptr ) {
            set_badlmids.insert(lmid);
            continue;
        }

        if( plm->isBad() ) {
            set_badlmids.insert(lmid);
            continue;
        }

        map_local_plms.emplace(lmid, plm);

        if( !pslamstate_->buse_inv_depth_ )
        {
            map_id_pointspar_.emplace(lmid, PointXYZParametersBlock(lmid, plm->getPoint()));

            problem.AddParameterBlock(map_id_pointspar_.at(lmid).values(), 3);
            ordering->AddElementToGroup(map_id_pointspar_.at(lmid).values(), 0);
        }

        int kfanchid = -1;
        double unanch_u = -1.;
        double unanch_v = -1.;

        for( const auto &kfid : plm->getKfObsSet() ) {

            if( kfid > newframe.kfid_ ) {
                continue;
            }

            auto pkfit = map_local_pkfs.find(kfid);
            std::shared_ptr<Frame> pkf = nullptr;

            // Add the observing KF if not set yet
            if( pkfit == map_local_pkfs.end() ) 
            {
                pkf = pmap_->getKeyframe(kfid);
                if( pkf == nullptr ) {
                    pmap_->removeMapPointObs(kfid,plm->lmid_);
                    continue;
                }
                map_local_pkfs.emplace(kfid, pkf);
                map_id_posespar_.emplace(kfid, PoseParametersBlock(kfid, pkf->getTwc()));

                ceres::LocalParameterization *local_parameterization = new SE3LeftParameterization();

                problem.AddParameterBlock(map_id_posespar_.at(kfid).values(), 7, local_parameterization);
                ordering->AddElementToGroup(map_id_posespar_.at(kfid).values(), 1);

                set_cstkfids.insert(kfid);
                problem.SetParameterBlockConstant(map_id_posespar_.at(kfid).values());
                
            } else {
                pkf = pkfit->second;
            }

            auto kp = pkf->getKeypointById(lmid);

            if( kp.lmid_ != lmid ) {
                pmap_->removeMapPointObs(lmid, kfid);
                continue;
            }

            if( pslamstate_->buse_inv_depth_ ) {
                if( kfanchid < 0 ) {
                    kfanchid = kfid;
                    unanch_u = kp.unpx_.x;
                    unanch_v = kp.unpx_.y;
                    double zanch = (pkf->getTcw() * plm->getPoint()).z();
                    map_id_invptspar_.emplace(lmid, InvDepthParametersBlock(lmid, kfanchid, zanch));

                    problem.AddParameterBlock(map_id_invptspar_.at(lmid).values(), 1);
                    ordering->AddElementToGroup(map_id_invptspar_.at(lmid).values(), 0);

                    if( kp.is_stereo_ ) 
                    {
                        ceres::CostFunction *f = new DirectLeftSE3::ReprojectionErrorRightAnchCamKSE3AnchInvDepth(
                                kp.runpx_.x, kp.runpx_.y, unanch_u, unanch_v, 
                                std::pow(2.,kp.scale_)
                            );

                        ceres::ResidualBlockId rid = problem.AddResidualBlock(
                                f, loss_function, 
                                calibpar.values(), rightcalibpar.values(),
                                rlextrinpose.values(), 
                                map_id_invptspar_.at(lmid).values()
                            );

                        vanchright_reprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,lmid))));
                        nbstereo++;
                    } else {
                        nbmono++;
                    }
                    continue;
                }
            }

            ceres::CostFunction *f;
            ceres::ResidualBlockId rid;

            // Add a visual factor between KF-MP nodes
            if( kp.is_stereo_ ) {
                if( pslamstate_->buse_inv_depth_ ) {
                    
                    f = new DirectLeftSE3::ReprojectionErrorKSE3AnchInvDepth(
                                kp.unpx_.x, kp.unpx_.y, unanch_u, unanch_v, 
                                std::pow(2.,kp.scale_)
                            );

                    rid = problem.AddResidualBlock(
                                f, loss_function, 
                                calibpar.values(),
                                map_id_posespar_.at(kfanchid).values(),
                                map_id_posespar_.at(kfid).values(), 
                                map_id_invptspar_.at(lmid).values()
                            );
                        
                    vreprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,kp.lmid_))));

                    f = new DirectLeftSE3::ReprojectionErrorRightCamKSE3AnchInvDepth(
                            kp.runpx_.x, kp.runpx_.y, unanch_u, unanch_v, 
                            std::pow(2.,kp.scale_)
                        );

                    rid = problem.AddResidualBlock(
                            f, loss_function, 
                            calibpar.values(), rightcalibpar.values(),
                            map_id_posespar_.at(kfanchid).values(), 
                            map_id_posespar_.at(kfid).values(), 
                            rlextrinpose.values(), 
                            map_id_invptspar_.at(lmid).values()
                        );
                    
                    vright_reprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,lmid))));
                    nbstereo++;
                    continue;

                } else {
                    f = new DirectLeftSE3::ReprojectionErrorKSE3XYZ(
                                kp.unpx_.x, kp.unpx_.y, std::pow(2.,kp.scale_)
                            );

                    rid = problem.AddResidualBlock(
                                f, loss_function, 
                                calibpar.values(),
                                map_id_posespar_.at(kfid).values(), 
                                map_id_pointspar_.at(lmid).values()
                            );
                        
                    vreprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,lmid))));

                    f = new DirectLeftSE3::ReprojectionErrorRightCamKSE3XYZ(
                            kp.runpx_.x, kp.runpx_.y, std::pow(2.,kp.scale_)
                        );

                    rid = problem.AddResidualBlock(
                            f, loss_function, 
                            rightcalibpar.values(),
                            map_id_posespar_.at(kfid).values(), 
                            rlextrinpose.values(), 
                            map_id_pointspar_.at(lmid).values()
                        );
                    
                    vright_reprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,lmid))));
                    nbstereo++;
                }
            } 
            else {
                if( pslamstate_->buse_inv_depth_ ) {
                    f = new DirectLeftSE3::ReprojectionErrorKSE3AnchInvDepth(
                                kp.unpx_.x, kp.unpx_.y, unanch_u, unanch_v, 
                                std::pow(2.,kp.scale_)
                            );

                    rid = problem.AddResidualBlock(
                                f, loss_function, 
                                calibpar.values(),
                                map_id_posespar_.at(kfanchid).values(),
                                map_id_posespar_.at(kfid).values(), 
                                map_id_invptspar_.at(lmid).values()
                            );
                }
                else {
                    f = new DirectLeftSE3::ReprojectionErrorKSE3XYZ(
                                kp.unpx_.x, kp.unpx_.y, std::pow(2.,kp.scale_)
                            );

                    rid = problem.AddResidualBlock(
                                f, loss_function, calibpar.values(), 
                                map_id_posespar_.at(kfid).values(), 
                                map_id_pointspar_.at(lmid).values()
                            );
                }

                vreprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,kp.lmid_))));

                nbmono++;
            }
        }
    }


    // Ensure the gauge is fixed
    size_t nbcstkfs = set_cstkfids.size();

    // At least two fixed KF in mono / one fixed KF in Stereo / RGB-D
    if( nbcstkfs < nmincstkfs  ) 
    {
        size_t nkf2add = nmincstkfs - nbcstkfs;
        size_t i = 0;
        for( auto it = map_id_posespar_.begin() ; i < nkf2add ; i++, it++ )
        {
            problem.SetParameterBlockConstant(map_id_posespar_.at(it->first).values());
            set_cstkfids.insert(it->first);
        }
    }
    
    nbcstkfs = set_cstkfids.size();
    size_t nbkfstot = map_local_pkfs.size();
    size_t nbkfs2opt = nbkfstot - nbcstkfs;
    size_t nblms2opt = map_local_plms.size();

    if( pslamstate_->debug_ ) {
        std::cout << "\n\n >>> [looseBA] problem setup!";
        std::cout << "\n >>> Kfs added (opt / tot) : " << nbkfs2opt 
            << " / " << nbkfstot;
        std::cout << "\n >>> MPs added : " << nblms2opt;
        std::cout << "\n >>> Measurements added (mono / stereo) : " 
            << nbmono << " / " << nbstereo;

        std::cout << "\n\n >>> looseBA KFs added : ";
        for( const auto &id_pkf : map_local_pkfs ) {
            std::cout << " KF #" << id_pkf.first << " (cst : " 
                << set_cstkfids.count(id_pkf.first) << "), ";
        }
    }

    // =================================
    //      Solve BA Problem
    // =================================
    
    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);

    if( pslamstate_->use_sparse_schur_ ) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
    } else {
        options.linear_solver_type = ceres::DENSE_SCHUR;
    }

    if( pslamstate_->use_dogleg_ ) {
        options.trust_region_strategy_type = ceres::DOGLEG;
        if( pslamstate_->use_subspace_dogleg_ ) {
            options.dogleg_type = ceres::DoglegType::SUBSPACE_DOGLEG;
        } else {
            options.dogleg_type = ceres::DoglegType::TRADITIONAL_DOGLEG;
        }
    } else {
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    }

    if( pslamstate_->use_nonmonotic_step_ ) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = true;
    }

    options.num_threads = 1;
    options.max_num_iterations = 5;
    options.function_tolerance = 1.e-4;
    
    options.minimizer_progress_to_stdout = pslamstate_->debug_;

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.LC_LooseBA_setup");

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.LC_LooseBA_Optimize");

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if( pslamstate_->debug_ )
        std::cout << summary.FullReport() << std::endl;

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.LC_LooseBA_Optimize");

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.LC_LooseBA_remove-outliers");


    // =================================
    //      Remove outliers
    // =================================

    size_t nbbadobsmono = 0;
    size_t nbbadobsrightcam = 0;

    // Remove Bad Observations
    std::vector<std::pair<int,int>> vbadkflmids;
    std::vector<std::pair<int,int>> vbadstereokflmids;
    vbadkflmids.reserve(vreprojerr_kfid_lmid.size() / 10);
    vbadstereokflmids.reserve(vright_reprojerr_kfid_lmid.size() / 10);

    for( auto it = vreprojerr_kfid_lmid.begin() 
        ; it != vreprojerr_kfid_lmid.end(); )
    {
        bool bbigchi2 = true;
        bool bdepthpos = true;
        if( pslamstate_->buse_inv_depth_ ) {
            auto *err = static_cast<DirectLeftSE3::ReprojectionErrorKSE3AnchInvDepth*>(it->first);
            bbigchi2 = err->chi2err_ > mono_th;
            bdepthpos = err->isdepthpositive_;
        }
        else {
            auto *err = static_cast<DirectLeftSE3::ReprojectionErrorKSE3XYZ*>(it->first);
            bbigchi2 = err->chi2err_ > mono_th;
            bdepthpos = err->isdepthpositive_;
        }
        
        if( bbigchi2 || !bdepthpos )
        {
            if( pslamstate_->apply_l2_after_robust_ ) {
                auto rid = it->second.first;
                problem.RemoveResidualBlock(rid);
            }
            int lmid = it->second.second.second;
            int kfid = it->second.second.first;
            vbadkflmids.push_back(std::pair<int,int>(kfid,lmid));
            set_badlmids.insert(lmid);
            nbbadobsmono++;

            it = vreprojerr_kfid_lmid.erase(it);
        } else {
            it++;
        }
    }

    for( auto it = vright_reprojerr_kfid_lmid.begin() 
    ; it != vright_reprojerr_kfid_lmid.end() ; )
    {
        bool bbigchi2 = true;
        bool bdepthpos = true;
        if( pslamstate_->buse_inv_depth_ ) {
            auto *err = static_cast<DirectLeftSE3::ReprojectionErrorRightCamKSE3AnchInvDepth*>(it->first);
            bbigchi2 = err->chi2err_ > mono_th;
            bdepthpos = err->isdepthpositive_;
        }
        else {
            auto *err = static_cast<DirectLeftSE3::ReprojectionErrorRightCamKSE3XYZ*>(it->first);
            bbigchi2 = err->chi2err_ > mono_th;
            bdepthpos = err->isdepthpositive_;
        }
        
        if( bbigchi2 || !bdepthpos )
        {
            if( pslamstate_->apply_l2_after_robust_ ) {
                auto rid = it->second.first;
                problem.RemoveResidualBlock(rid);
            }
            int lmid = it->second.second.second;
            int kfid = it->second.second.first;
            vbadstereokflmids.push_back(std::pair<int,int>(kfid,lmid));
            set_badlmids.insert(lmid);
            nbbadobsrightcam++;

            it = vright_reprojerr_kfid_lmid.erase(it);
        } else {
            it++;
        }
    }

    for( auto it = vanchright_reprojerr_kfid_lmid.begin() 
    ; it != vanchright_reprojerr_kfid_lmid.end() ; )
    {
        bool bbigchi2 = true;
        bool bdepthpos = true;
        auto *err = static_cast<DirectLeftSE3::ReprojectionErrorRightAnchCamKSE3AnchInvDepth*>(it->first);
        bbigchi2 = err->chi2err_ > mono_th;
        bdepthpos = err->isdepthpositive_;

        if( bbigchi2 || !bdepthpos )
        {
            if( pslamstate_->apply_l2_after_robust_ ) {
                auto rid = it->second.first;
                problem.RemoveResidualBlock(rid);
            }
            int lmid = it->second.second.second;
            int kfid = it->second.second.first;
            vbadstereokflmids.push_back(std::pair<int,int>(kfid,lmid));
            set_badlmids.insert(lmid);
            nbbadobsrightcam++;

            it = vanchright_reprojerr_kfid_lmid.erase(it);
        } else {
            it++;
        }
    }


    // =================================
    //      Update State Parameters
    // =================================

    std::vector<int> vlmids, vkfids;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vwpt;
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vTwc;
    vlmids.reserve(nblms2opt);
    vwpt.reserve(nblms2opt);
    vkfids.reserve(nbkfs2opt);
    vTwc.reserve(nbkfs2opt);

    Sophus::SE3d iniTnewkfw = newframe.getTcw();
    Sophus::SE3d optTwnewkf = map_id_posespar_.at(newframe.kfid_).getPose();

    if( pslamstate_->debug_ ) {
        std::cout << "\n \t>>> looseBA() --> Kf loop old pos : " 
            << iniTnewkfw.inverse().translation().transpose() << "\n";
        std::cout << "\n \t>>> looseBA() --> Kf loop new pos : " 
            << optTwnewkf.translation().transpose() << "\n";
    }

    // Get Updated KFs
    for( const auto &kfid_pkf : map_local_pkfs )
    {
        int kfid = kfid_pkf.first;

        if( set_cstkfids.count(kfid) ) {
            continue;
        }
        
        auto pkf = kfid_pkf.second;

        if( pkf == nullptr ) {
            continue;
        }

        vkfids.push_back(kfid);
        vTwc.push_back(map_id_posespar_.at(kfid).getPose());
    }

    // Get Updated MPs
    for( const auto &lmid_plm : map_local_plms )
    {
        int lmid = lmid_plm.first;
        auto plm = lmid_plm.second;

        if( plm == nullptr ) {
            continue;
        }

        if( plm->isBad() ) {
            set_badlmids.insert(lmid);
            continue;
        } 

        if( pslamstate_->buse_inv_depth_ ) {
            auto invptit = map_id_invptspar_.find(lmid);
            if( invptit == map_id_invptspar_.end() ) {
                set_badlmids.insert(lmid);
                continue;
            }
            double zanch = 1. / invptit->second.getInvDepth();
            if( zanch <= 0. ) {
                set_badlmids.insert(lmid);
                continue;
            }

            auto it = map_local_pkfs.find(plm->kfid_);
            if( it == map_local_pkfs.end() ) {
                set_badlmids.insert(lmid);
                continue;
            }
            auto pkfanch = it->second;
            
            if( pkfanch != nullptr ) {
                auto kp = pkfanch->getKeypointById(lmid);
                Eigen::Vector3d uvpt(kp.unpx_.x,kp.unpx_.y,1.); 
                Eigen::Vector3d optwpt = pkfanch->getTwc() * (zanch * pkfanch->pcalib_leftcam_->iK_ * uvpt);
                vlmids.push_back(lmid);
                vwpt.push_back(optwpt);
            } else {
                set_badlmids.insert(lmid);
            }
        } else {
            auto optlmit = map_id_pointspar_.find(lmid);
            if( optlmit != map_id_pointspar_.end() ) {
                vlmids.push_back(lmid);
                vwpt.push_back(optlmit->second.getPoint());
            } else {
                set_badlmids.insert(lmid);
            }
        }
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.LC_LooseBA_remove-outliers");

    std::lock_guard<std::mutex> lock2(pmap_->optim_mutex_);

    std::lock_guard<std::mutex> lock(pmap_->map_mutex_);

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.LC_LooseBA_Update");

    // Update KFs / MPs
    for( size_t i = 0, iend = vlmids.size() ; i < iend ; i++ )
    {
        int lmid = vlmids.at(i);
        pmap_->updateMapPoint(lmid, vwpt.at(i));
    }

    for( size_t i = 0, iend = vkfids.size() ; i < iend ; i++ )
    {
        int kfid = vkfids.at(i);
        auto pkf = pmap_->getKeyframe(kfid);
        if( pkf != nullptr ) {
            pkf->setTwc(vTwc.at(i));
        }
    }

    std::unordered_set<int> uplmid_set;

    // Propagate Corrections to youngest KFs / MPs
    for( int kfid = newframe.kfid_+1 ; kfid <= pmap_->nkfid_ ; kfid++ )
    {
        if( map_local_pkfs.count(kfid) ) {
            continue;
        }

        auto pkf = pmap_->getKeyframe(kfid);
        if( pkf == nullptr ) {
            continue;
        }

        Sophus::SE3d prevTwkf = pkf->getTwc();
        Sophus::SE3d relTnewkf_kf = iniTnewkfw * prevTwkf;

        Sophus::SE3d updTwkf = optTwnewkf * relTnewkf_kf;

        for( const auto & kp : pkf->getKeypoints3d() ) 
        {
            if( uplmid_set.count(kp.lmid_) ) {
                continue;
            }

            if( map_local_plms.count(kp.lmid_) ) {
                continue;
            }

            auto plm = pmap_->getMapPoint(kp.lmid_);
            if( plm == nullptr ) {
                pmap_->removeMapPointObs(kp.lmid_, kfid);
                continue;
            }
            
            if( plm->kfid_ == kfid ) {
                Eigen::Vector3d campt = pkf->projWorldToCam(plm->getPoint());
                pmap_->updateMapPoint(kp.lmid_, updTwkf * campt);
                uplmid_set.insert(plm->lmid_);
            }
        }

        pkf->setTwc(updTwkf);
    }

    // Remove Bad obs
    for( const auto &badkflmid : vbadstereokflmids ) {
        int kfid = badkflmid.first;
        int lmid = badkflmid.second;
        auto it = map_local_pkfs.find(kfid);
        if( it != map_local_pkfs.end() ) {
            it->second->removeStereoKeypointById(lmid);
        }
        set_badlmids.insert(lmid);
    }

    for( const auto &badkflmid : vbadkflmids ) {
        int kfid = badkflmid.first;
        int lmid = badkflmid.second;
        auto it = map_local_pkfs.find(kfid);
        if( it != map_local_pkfs.end() ) {
            pmap_->removeMapPointObs(lmid,kfid);
        }
        if( kfid == pmap_->pcurframe_->kfid_ ) {
            pmap_->removeObsFromCurFrameById(lmid);
        }
        set_badlmids.insert(lmid);
    }


    // Map Point Culling for bad Obs.
    size_t nbbadlm = 0;
    for( const auto &lmid : set_badlmids ) {

        std::shared_ptr<MapPoint> plm;
        auto plmit = map_local_plms.find(lmid);
        if( plmit == map_local_plms.end() ) {
            plm = pmap_->getMapPoint(lmid);
        } else {
            plm = plmit->second;
        }
        if( plm == nullptr ) {
            continue;
        }
        
        if( plm->isBad() ) {
            pmap_->removeMapPoint(lmid);
            nbbadlm++;
        } else {
            auto set_cokfs = plm->getKfObsSet();
            if( set_cokfs.size() < 3 ) {
                if( plm->kfid_ < newframe.kfid_-3 && !plm->isobs_ ) {
                    pmap_->removeMapPoint(lmid);
                    nbbadlm++;
                }
            }
        }
    }

    // Update cur frame pose
    Sophus::SE3d prevTwcur = pmap_->pcurframe_->getTwc();
    Sophus::SE3d relTlckf_cur = iniTnewkfw * prevTwcur;

    Sophus::SE3d updTwcur = optTwnewkf * relTlckf_cur;

    pmap_->pcurframe_->setTwc(updTwcur);

    size_t nbbadobs = nbbadobsmono + nbbadobsrightcam;

    if( pslamstate_->debug_ ) {
        std::cout << "\n \t>>> looseBA() --> Nb of bad obs / nb removed MP : " 
            << nbbadobs << " / " << nbbadlm;
        std::cout << "\n \t>>> looseBA() --> Nb of bad obs mono / stereo / right : " 
            << nbbadobsmono << " / " << nbbadobsrightcam;
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.LC_LooseBA_Update");
}



void Optimizer::fullBA(const bool buse_robust_cost)
{
    // =================================
    //      Setup BA Problem
    // =================================

    Frame &newframe = *pmap_->getKeyframe(0);

    ceres::Problem problem;
    ceres::LossFunctionWrapper *loss_function;
    
    const float mono_th = pslamstate_->robust_mono_th_;
    
    auto *mono_robust_loss =  new ceres::HuberLoss(std::sqrt(mono_th));

    loss_function = new ceres::LossFunctionWrapper(mono_robust_loss, ceres::TAKE_OWNERSHIP);

    if( !buse_robust_cost ) {
        loss_function->Reset(nullptr, ceres::TAKE_OWNERSHIP);
    }

    size_t nmincstkfs = 2;

    if( pslamstate_->stereo_ ) {
        nmincstkfs = 1;
    }

    size_t nbmono = 0;
    size_t nbstereo = 0;

    auto ordering = new ceres::ParameterBlockOrdering;

    std::unordered_map<int, PoseParametersBlock> map_id_posespar_;
    std::unordered_map<int, PointXYZParametersBlock> map_id_pointspar_;
    std::unordered_map<int, InvDepthParametersBlock> map_id_invptspar_;

    std::unordered_map<int, std::shared_ptr<MapPoint>> map_local_plms;
    std::unordered_map<int, std::shared_ptr<Frame>> map_local_pkfs;

    // Storing the factors and their residuals block ids 
    // for fast accessing when cheking for outliers
    std::vector<
        std::pair<ceres::CostFunction*, 
        std::pair<ceres::ResidualBlockId, 
            std::pair<int,int>
    >>> 
        vreprojerr_kfid_lmid, vright_reprojerr_kfid_lmid;

    // Add the left cam calib parameters
    auto pcalibleft = newframe.pcalib_leftcam_;
    CalibParametersBlock calibpar(0, pcalibleft->fx_, pcalibleft->fy_, pcalibleft->cx_, pcalibleft->cy_);
    problem.AddParameterBlock(calibpar.values(), 4);
    ordering->AddElementToGroup(calibpar.values(), 1);

    problem.SetParameterBlockConstant(calibpar.values());
    
    // Prepare variables if STEREO mode
    auto pcalibright = newframe.pcalib_rightcam_;
    CalibParametersBlock rightcalibpar;
    
    Sophus::SE3d Trl, Tlr;
    PoseParametersBlock rlextrinpose(0, Trl);

    if( pslamstate_->stereo_ ) {
        // Right Intrinsic
        rightcalibpar = CalibParametersBlock(0, pcalibright->fx_, pcalibright->fy_, pcalibright->cx_, pcalibright->cy_);
        problem.AddParameterBlock(rightcalibpar.values(), 4);
        ordering->AddElementToGroup(rightcalibpar.values(), 1);

        problem.SetParameterBlockConstant(rightcalibpar.values());

        // Right Extrinsic
        Tlr = pcalibright->getExtrinsic();
        Trl = Tlr.inverse();
        rlextrinpose = PoseParametersBlock(0, Trl);

        ceres::LocalParameterization *local_param = new SE3LeftParameterization();

        problem.AddParameterBlock(rlextrinpose.values(), 7, local_param);
        ordering->AddElementToGroup(rlextrinpose.values(), 1);

        problem.SetParameterBlockConstant(rlextrinpose.values());
    }

    // Keep track of MPs no suited for BA for speed-up
    std::unordered_set<int> set_badlmids;

    std::set<int> set_kfids2opt;
    std::set<int> set_lmids2opt;
    std::set<int> set_cstkfids;

    if( pslamstate_->debug_ )
        std::cout << "\n >>> Full BA \n";

    // Add the KFs to optimize to BA (new KF + cov KFs)
    for( int kfid = 0 ; kfid <= pmap_->nkfid_ ; kfid++ )
    {
        // Add every KF to BA problem
        auto pkf = pmap_->getKeyframe(kfid);

        if( pkf == nullptr ) {
            continue;
        }

        map_id_posespar_.emplace(pkf->kfid_, PoseParametersBlock(pkf->kfid_, pkf->getTwc()));

        ceres::LocalParameterization *local_parameterization = new SE3LeftParameterization();

        problem.AddParameterBlock(map_id_posespar_.at(pkf->kfid_).values(), 7, local_parameterization);
        ordering->AddElementToGroup(map_id_posespar_.at(pkf->kfid_).values(), 1);

        if( set_cstkfids.size() < nmincstkfs ) {
            set_cstkfids.insert(pkf->kfid_);
            problem.SetParameterBlockConstant(map_id_posespar_.at(pkf->kfid_).values());
        } else {
            // Add observed MPs for KFs' to optimize
            set_kfids2opt.insert(pkf->kfid_);
            for( const auto &kp : pkf->getKeypoints3d() ) {
                set_lmids2opt.insert(kp.lmid_);
            }
        }

        map_local_pkfs.emplace(pkf->kfid_, pkf);
    }

    // Go through the MPs to optimize
    for( const auto &lmid : set_lmids2opt ) {
        auto plm = pmap_->getMapPoint(lmid);

        if( plm == nullptr ) {
            set_badlmids.insert(lmid);
            continue;
        }

        if( plm->isBad() ) {
            set_badlmids.insert(lmid);
            continue;
        }

        if( plm->getKfObsSet().size() < 3 ) {
            set_badlmids.insert(lmid);
            continue;
        }

        map_local_plms.emplace(lmid, plm);

        if( !pslamstate_->buse_inv_depth_ )
        {
            map_id_pointspar_.emplace(lmid, PointXYZParametersBlock(lmid, plm->getPoint()));

            problem.AddParameterBlock(map_id_pointspar_.at(lmid).values(), 3);
            ordering->AddElementToGroup(map_id_pointspar_.at(lmid).values(), 0);
        }

        int kfanchid = -1;
        double unanch_u = -1.;
        double unanch_v = -1.;

        for( const auto &kfid : plm->getKfObsSet() ) {

            auto pkfit = map_local_pkfs.find(kfid);
            std::shared_ptr<Frame> pkf = nullptr;

            // Add the observing KF if not set yet
            if( pkfit == map_local_pkfs.end() ) 
            {
                pkf = pmap_->getKeyframe(kfid);
                if( pkf == nullptr ) {
                    pmap_->removeMapPointObs(kfid,plm->lmid_);
                    continue;
                }
                map_local_pkfs.emplace(kfid, pkf);
                map_id_posespar_.emplace(kfid, PoseParametersBlock(kfid, pkf->getTwc()));

                ceres::LocalParameterization *local_parameterization = new SE3LeftParameterization();

                problem.AddParameterBlock(map_id_posespar_.at(kfid).values(), 7, local_parameterization);
                ordering->AddElementToGroup(map_id_posespar_.at(kfid).values(), 1);

                set_cstkfids.insert(kfid);
                problem.SetParameterBlockConstant(map_id_posespar_.at(kfid).values());
                continue;
            } else {
                pkf = pkfit->second;
            }

            auto kp = pkf->getKeypointById(lmid);

            if( kp.lmid_ < 0 ) {
                pmap_->removeMapPointObs(lmid, kfid);
                continue;
            }

            if( pslamstate_->buse_inv_depth_ ) {
                if( kfanchid < 0 ) {
                    kfanchid = kfid;
                    unanch_u = kp.unpx_.x;
                    unanch_v = kp.unpx_.y;
                    double zanch = (pkf->getTcw() * plm->getPoint()).z();
                    map_id_invptspar_.emplace(lmid, InvDepthParametersBlock(lmid, kfanchid, zanch));

                    problem.AddParameterBlock(map_id_invptspar_.at(lmid).values(), 1);
                    ordering->AddElementToGroup(map_id_invptspar_.at(lmid).values(), 0);

                    if( kp.is_stereo_ ) 
                    {
                        ceres::CostFunction *f = new DirectLeftSE3::ReprojectionErrorRightAnchCamKSE3AnchInvDepth(
                                kp.runpx_.x, kp.runpx_.y, unanch_u, unanch_v, 
                                std::pow(2.,kp.scale_)
                            );

                        ceres::ResidualBlockId rid = problem.AddResidualBlock(
                                f, loss_function, 
                                calibpar.values(), rightcalibpar.values(),
                                rlextrinpose.values(), 
                                map_id_invptspar_.at(lmid).values()
                            );

                        vright_reprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,lmid))));
                    }
                    continue;
                }
            }

            ceres::CostFunction *f;
            ceres::ResidualBlockId rid;

            // Add a visual factor between KF-MP nodes
            if( kp.is_stereo_ ) {
                if( pslamstate_->buse_inv_depth_ ) {
                    f = new DirectLeftSE3::ReprojectionErrorKSE3AnchInvDepth(
                                kp.unpx_.x, kp.unpx_.y, unanch_u, unanch_v, 
                                std::pow(2.,kp.scale_)
                            );

                    rid = problem.AddResidualBlock(
                                f, loss_function, 
                                calibpar.values(),
                                map_id_posespar_.at(kfanchid).values(),
                                map_id_posespar_.at(kfid).values(), 
                                map_id_invptspar_.at(lmid).values()
                            );
                        
                    vreprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,kp.lmid_))));

                    f = new DirectLeftSE3::ReprojectionErrorRightCamKSE3AnchInvDepth(
                            kp.runpx_.x, kp.runpx_.y, unanch_u, unanch_v, 
                            std::pow(2.,kp.scale_)
                        );

                    rid = problem.AddResidualBlock(
                            f, loss_function, 
                            calibpar.values(), rightcalibpar.values(),
                            map_id_posespar_.at(kfanchid).values(), 
                            map_id_posespar_.at(kfid).values(), 
                            rlextrinpose.values(), 
                            map_id_invptspar_.at(lmid).values()
                        );
                    
                    vright_reprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,lmid))));
                    nbstereo++;
                    continue;
                } else {
                    f = new DirectLeftSE3::ReprojectionErrorKSE3XYZ(
                                kp.unpx_.x, kp.unpx_.y, std::pow(2.,kp.scale_)
                            );

                    rid = problem.AddResidualBlock(
                                f, loss_function, 
                                calibpar.values(),
                                map_id_posespar_.at(kfid).values(), 
                                map_id_pointspar_.at(lmid).values()
                            );
                        
                    vreprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,lmid))));

                    f = new DirectLeftSE3::ReprojectionErrorRightCamKSE3XYZ(
                            kp.runpx_.x, kp.runpx_.y, std::pow(2.,kp.scale_)
                        );

                    rid = problem.AddResidualBlock(
                            f, loss_function, 
                            rightcalibpar.values(),
                            map_id_posespar_.at(kfid).values(), 
                            rlextrinpose.values(), 
                            map_id_pointspar_.at(lmid).values()
                        );
                    
                    vright_reprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,lmid))));
                    nbstereo++;
                }
            } 
            else {
                if( pslamstate_->buse_inv_depth_ ) {
                    f = new DirectLeftSE3::ReprojectionErrorKSE3AnchInvDepth(
                                kp.unpx_.x, kp.unpx_.y, unanch_u, unanch_v, 
                                std::pow(2.,kp.scale_)
                            );

                    rid = problem.AddResidualBlock(
                                f, loss_function, 
                                calibpar.values(),
                                map_id_posespar_.at(kfanchid).values(),
                                map_id_posespar_.at(kfid).values(), 
                                map_id_invptspar_.at(lmid).values()
                            );
                }
                else {
                    f = new DirectLeftSE3::ReprojectionErrorKSE3XYZ(
                                kp.unpx_.x, kp.unpx_.y, std::pow(2.,kp.scale_)
                            );

                    rid = problem.AddResidualBlock(
                                f, loss_function, calibpar.values(), 
                                map_id_posespar_.at(kfid).values(), 
                                map_id_pointspar_.at(lmid).values()
                            );
                }

                vreprojerr_kfid_lmid.push_back(std::make_pair(f, std::make_pair(rid, std::make_pair(kfid,kp.lmid_))));

                nbmono++;
            }
        }
    }


    // Ensure the gauge is fixed
    size_t nbcstkfs = set_cstkfids.size();

    // At least two fixed KF in mono / one fixed KF in Stereo / RGB-D
    if( nbcstkfs < nmincstkfs  ) 
    {
        size_t nkf2add = nmincstkfs - nbcstkfs;
        size_t i = 0;
        for( auto it = map_id_posespar_.begin() ; i < nkf2add ; i++, it++ )
        {
            problem.SetParameterBlockConstant(map_id_posespar_.at(it->first).values());
            set_cstkfids.insert(it->first);
        }
    }
    
    nbcstkfs = set_cstkfids.size();
    size_t nbkfstot = map_local_pkfs.size();
    size_t nbkfs2opt = nbkfstot - nbcstkfs;
    size_t nblms2opt = map_local_plms.size();


    if( pslamstate_->debug_ ) {
        std::cout << "\n >>> FullBA problem setup!";
        std::cout << "\n >>> Kfs added (opt / tot) : " << nbkfs2opt 
            << " / " << nbkfstot;
        std::cout << "\n >>> MPs added : " << nblms2opt;
    }

    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);

    if( pslamstate_->use_sparse_schur_ ) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
    } else {
        options.linear_solver_type = ceres::DENSE_SCHUR;
    }

    if( pslamstate_->use_dogleg_ ) {
        options.trust_region_strategy_type = ceres::DOGLEG;
        if( pslamstate_->use_subspace_dogleg_ ) {
            options.dogleg_type = ceres::DoglegType::SUBSPACE_DOGLEG;
        } else {
            options.dogleg_type = ceres::DoglegType::TRADITIONAL_DOGLEG;
        }
    } else {
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    }

    if( pslamstate_->use_nonmonotic_step_ ) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = true;
    }

    options.num_threads = 8;
    options.max_num_iterations = 100;
    
    options.minimizer_progress_to_stdout = pslamstate_->debug_;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if( pslamstate_->debug_ )
        std::cout << summary.FullReport() << std::endl;


    size_t nbbadobsmono = 0;
    size_t nbbadobsrightcam = 0;

    for( auto it = vreprojerr_kfid_lmid.begin() 
        ; it != vreprojerr_kfid_lmid.end(); )
    {
        bool bbigchi2 = true;
        bool bdepthpos = true;
        if( pslamstate_->buse_inv_depth_ ) {
            auto *err = static_cast<DirectLeftSE3::ReprojectionErrorKSE3AnchInvDepth*>(it->first);
            bbigchi2 = err->chi2err_ > mono_th;
            bdepthpos = err->isdepthpositive_;
        }
        else {
            auto *err = static_cast<DirectLeftSE3::ReprojectionErrorKSE3XYZ*>(it->first);
            bbigchi2 = err->chi2err_ > mono_th;
            bdepthpos = err->isdepthpositive_;
        }
        
        if( bbigchi2 || !bdepthpos )
        {
            if( pslamstate_->apply_l2_after_robust_ ) {
                auto rid = it->second.first;
                problem.RemoveResidualBlock(rid);
            }
            int lmid = it->second.second.second;
            int kfid = it->second.second.first;
            pmap_->removeMapPointObs(lmid,kfid);
            set_badlmids.insert(lmid);
            nbbadobsmono++;

            it = vreprojerr_kfid_lmid.erase(it);
        } else {
            it++;
        }
    }

    if( !vright_reprojerr_kfid_lmid.empty() ) {
        for( auto it = vright_reprojerr_kfid_lmid.begin() 
        ; it != vright_reprojerr_kfid_lmid.end(); )
        {
            bool bbigchi2 = true;
            bool bdepthpos = true;
            if( pslamstate_->buse_inv_depth_ ) {
                auto *err = static_cast<DirectLeftSE3::ReprojectionErrorRightCamKSE3AnchInvDepth*>(it->first);
                bbigchi2 = err->chi2err_ > mono_th;
                bdepthpos = err->isdepthpositive_;
            }
            else {
                auto *err = static_cast<DirectLeftSE3::ReprojectionErrorRightCamKSE3XYZ*>(it->first);
                bbigchi2 = err->chi2err_ > mono_th;
                bdepthpos = err->isdepthpositive_;
            }

            if( bbigchi2 || !bdepthpos )
            {
                if( pslamstate_->apply_l2_after_robust_ ) {
                    auto rid = it->second.first;
                    problem.RemoveResidualBlock(rid);
                }
                int lmid = it->second.second.second;
                int kfid = it->second.second.first;
                map_local_pkfs.at(kfid)->removeStereoKeypointById(lmid);
                set_badlmids.insert(lmid);
                nbbadobsrightcam++;

                it = vright_reprojerr_kfid_lmid.erase(it);
            } else {
                it++;
            }
        }
    }

    size_t nbbadobs = nbbadobsmono + nbbadobsrightcam;

    // Refine without Robust cost
    if( pslamstate_->apply_l2_after_robust_ && nbbadobs > 0 ) 
    {
        if( !vreprojerr_kfid_lmid.empty() ) {
            loss_function->Reset(nullptr, ceres::TAKE_OWNERSHIP);
        }

        ceres::Solve(options, &problem, &summary);

        if( pslamstate_->debug_ )
            std::cout << summary.FullReport() << std::endl;
    }

    nbbadobsmono = 0;
    nbbadobsrightcam = 0;

    for( auto it = vreprojerr_kfid_lmid.begin() 
        ; it != vreprojerr_kfid_lmid.end(); )
    {
        bool bbigchi2 = true;
        bool bdepthpos = true;
        if( pslamstate_->buse_inv_depth_ ) {
            auto *err = static_cast<DirectLeftSE3::ReprojectionErrorKSE3AnchInvDepth*>(it->first);
            bbigchi2 = err->chi2err_ > mono_th;
            bdepthpos = err->isdepthpositive_;
        }
        else {
            auto *err = static_cast<DirectLeftSE3::ReprojectionErrorKSE3XYZ*>(it->first);
            bbigchi2 = err->chi2err_ > mono_th;
            bdepthpos = err->isdepthpositive_;
        }
        
        if( bbigchi2 || !bdepthpos )
        {
            if( pslamstate_->apply_l2_after_robust_ ) {
                auto rid = it->second.first;
                problem.RemoveResidualBlock(rid);
            }
            int lmid = it->second.second.second;
            int kfid = it->second.second.first;
            pmap_->removeMapPointObs(lmid,kfid);
            set_badlmids.insert(lmid);
            nbbadobsmono++;

            it = vreprojerr_kfid_lmid.erase(it);
        } else {
            it++;
        }
    }

    if( !vright_reprojerr_kfid_lmid.empty() ) {
        for( auto it = vright_reprojerr_kfid_lmid.begin() 
        ; it != vright_reprojerr_kfid_lmid.end(); )
        {
            bool bbigchi2 = true;
            bool bdepthpos = true;
            if( pslamstate_->buse_inv_depth_ ) {
                auto *err = static_cast<DirectLeftSE3::ReprojectionErrorRightCamKSE3AnchInvDepth*>(it->first);
                bbigchi2 = err->chi2err_ > mono_th;
                bdepthpos = err->isdepthpositive_;
            }
            else {
                auto *err = static_cast<DirectLeftSE3::ReprojectionErrorRightCamKSE3XYZ*>(it->first);
                bbigchi2 = err->chi2err_ > mono_th;
                bdepthpos = err->isdepthpositive_;
            }

            if( bbigchi2 || !bdepthpos )
            {
                if( pslamstate_->apply_l2_after_robust_ ) {
                    auto rid = it->second.first;
                    problem.RemoveResidualBlock(rid);
                }
                int lmid = it->second.second.second;
                int kfid = it->second.second.first;
                map_local_pkfs.at(kfid)->removeStereoKeypointById(lmid);
                set_badlmids.insert(lmid);
                nbbadobsrightcam++;

                it = vright_reprojerr_kfid_lmid.erase(it);
            } else {
                it++;
            }
        }
    }
    
    // Update KFs
    for( const auto &kfid_pkf : map_local_pkfs )
    {
        int kfid = kfid_pkf.first;

        if( set_cstkfids.count(kfid) ) {
            continue;
        }
        
        auto pkf = kfid_pkf.second;

        if( pkf == nullptr ) {
            continue;
        }

        auto optkfpose = map_id_posespar_.at(kfid);
        pkf->setTwc(optkfpose.getPose());
    }

    // Update MPs
    for( const auto &lmid_plm : map_local_plms )
    {
        int lmid = lmid_plm.first;
        auto plm = lmid_plm.second;

        if( plm == nullptr ) {
            continue;
        }

        // Map Point Culling
        auto kfids = plm->getKfObsSet();
        if( kfids.size() < 3 ) {
            if( plm->kfid_ < newframe.kfid_-3 ) {
                pmap_->removeMapPoint(lmid);
                set_badlmids.erase(lmid);
                continue;
            }
        }

        if( pslamstate_->buse_inv_depth_ ) {
            auto invptit = map_id_invptspar_.find(lmid);
            if( invptit == map_id_invptspar_.end() ) {
                set_badlmids.erase(lmid);
                continue;
            }
            double zanch = 1. / invptit->second.getInvDepth();
            if( zanch <= 0. ) {
                pmap_->removeMapPoint(lmid);
                set_badlmids.erase(lmid);
                continue;
            }

            auto it = map_local_pkfs.find(plm->kfid_);
            if( it == map_local_pkfs.end() ) {
                set_badlmids.insert(lmid);
                continue;
            }
            auto pkfanch = it->second;
            
            if( pkfanch != nullptr ) {
                auto kp = pkfanch->getKeypointById(lmid);
                Eigen::Vector3d uvpt(kp.unpx_.x,kp.unpx_.y,1.); 
                Eigen::Vector3d optwpt = pkfanch->getTwc() * (zanch * pkfanch->pcalib_leftcam_->iK_ * uvpt);
                pmap_->updateMapPoint(lmid, optwpt, invptit->second.getInvDepth());
            } else {
                set_badlmids.insert(lmid);
            }
        } else {
            auto optlmit = map_id_pointspar_.find(lmid);
            if( optlmit != map_id_pointspar_.end() ) {
                pmap_->updateMapPoint(lmid, optlmit->second.getPoint());
            } else {
                set_badlmids.insert(lmid);
            }
        }
    }

    // Map Point Culling for bad Obs.

    for( const auto &lmid : set_badlmids ) {
        // auto plm = pmap_->getMapPoint(lmid);
        std::shared_ptr<MapPoint> plm;
        auto plmit = map_local_plms.find(lmid);
        if( plmit == map_local_plms.end() ) {
            plm = pmap_->getMapPoint(lmid);
        } else {
            plm = plmit->second;
        }
        if( plm == nullptr ) {
            continue;
        }
        
        if( plm->isBad() ) {
            pmap_->removeMapPoint(lmid);
        } else {
            auto set_cokfs = plm->getKfObsSet();
            if( set_cokfs.size() < 3 ) {
                if( plm->kfid_ < newframe.kfid_-3 ) {
                    pmap_->removeMapPoint(lmid);
                }
            }
        }
    }
}


void Optimizer::signalStopLocalBA()
{
    std::lock_guard<std::mutex> lock(localba_mutex_);
    bstop_localba_ = true;
}

bool Optimizer::stopLocalBA()
{
    std::lock_guard<std::mutex> lock(localba_mutex_);
    return bstop_localba_;
}

bool Optimizer::localPoseGraph(Frame &newframe, int kfloop_id, const Sophus::SE3d& newTwc)
{

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.LC_PoseGraph_setup");

    ceres::Problem problem;
    
    std::map<int, PoseParametersBlock> map_id_posespar_;
    std::map<int, std::shared_ptr<Frame>> map_pkfs;

    if( pslamstate_->debug_ ) 
        std::cout << "\n Going to opt pose graph between KF #" << kfloop_id 
            << " and KF #" << newframe.kfid_ << "\n";

    Sophus::SE3d iniTcw = newframe.getTcw();

    if( pslamstate_->debug_ )
        std::cout << "\n Adding loop KF : ";

    auto ploopkf = pmap_->getKeyframe(kfloop_id);

    while( ploopkf == nullptr ) {
        kfloop_id++;
        ploopkf = pmap_->getKeyframe(kfloop_id);
    }
    
    Sophus::SE3d Twc = ploopkf->getTwc();
    map_id_posespar_.emplace(kfloop_id, PoseParametersBlock(kfloop_id, Twc));

    ceres::LocalParameterization *local_parameterization = new SE3LeftParameterization();

    problem.AddParameterBlock(map_id_posespar_.at(kfloop_id).values(), 7, local_parameterization);

    problem.SetParameterBlockConstant(map_id_posespar_.at(kfloop_id).values());

    Sophus::SE3d Tciw = ploopkf->getTcw();
    int ci_id = kfloop_id;

    Sophus::SE3d Tloop_new = Tciw * newTwc;

    for( int kfid = kfloop_id+1 ; kfid <= newframe.kfid_ ; kfid++ ) {

        auto pkf = pmap_->getKeyframe(kfid);

        if( pkf == nullptr ) {
            if( kfid == newframe.kfid_ ) {
                return false;
            } else {
                continue;
            }
        }

        if( pslamstate_->debug_ )
            std::cout << kfid << ", ";

        map_pkfs.emplace(kfid, pkf);

        Sophus::SE3d Twcj = pkf->getTwc();

        map_id_posespar_.emplace(kfid, PoseParametersBlock(kfid, Twcj));

        ceres::LocalParameterization *local_parameterization = new SE3LeftParameterization();
        problem.AddParameterBlock(map_id_posespar_.at(kfid).values(), 7, local_parameterization);

        Sophus::SE3d Tcicj = Tciw * Twcj;

        ceres::CostFunction* f = new LeftSE3RelativePoseError(Tcicj);
        
        problem.AddResidualBlock(f, nullptr, map_id_posespar_.at(ci_id).values(), map_id_posespar_.at(kfid).values());
        
        Tciw = Twcj.inverse();
        ci_id = kfid;
    }

    ceres::CostFunction* f
        = new LeftSE3RelativePoseError(Tloop_new);
    
    problem.AddResidualBlock(f, nullptr, map_id_posespar_.at(kfloop_id).values(), map_id_posespar_.at(newframe.kfid_).values());

    Sophus::SE3d Tcurloop = newframe.getTcw() * ploopkf->getTwc();

    Eigen::Matrix<double,6,1> verr = (Tcurloop * Tloop_new).log();

    if( pslamstate_->debug_ )
        std::cout << "\n Loop Error : " << verr.norm() << " / " 
            << verr.transpose() << "\n";


    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.LC_PoseGraph_setup");

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.LC_PoseGraph_Optimize");

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    options.max_num_iterations = 10;
    options.function_tolerance = 1.e-4;
    
    options.minimizer_progress_to_stdout = pslamstate_->debug_;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if( pslamstate_->debug_ )
        std::cout << summary.FullReport() << std::endl;

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.LC_PoseGraph_Optimize");

    auto newkfpose = map_id_posespar_.at(newframe.kfid_);
    Sophus::SE3d newoptTwc = newkfpose.getPose();

    if( pslamstate_->debug_ ) {
        std::cout << "\nLC p3p pos : " << newTwc.translation().transpose() << "\n";
        std::cout << "\nLC opt pos : " << newoptTwc.translation().transpose() << "\n";
    }

    if( (newTwc.translation() - newoptTwc.translation()).norm() > 0.3
            && pslamstate_->stereo_ ) 
    {
        if( pslamstate_->debug_ )
            std::cout << "\n [PoseGraph] Skipping as we are most likely with a degenerate solution!";

        return false;
    }

    std::unordered_set<int> processed_lmids;

    std::vector<int> vlmids, vkfids;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vwpt;
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vTwc;
    vlmids.reserve(pmap_->nblms_);
    vwpt.reserve(pmap_->nblms_);
    vkfids.reserve(pmap_->nbkfs_);
    vTwc.reserve(pmap_->nbkfs_);

    // Get updated KFs / MPs
    for( const auto &kfid_pkf : map_pkfs )
    {
        int kfid = kfid_pkf.first;
        auto pkf = kfid_pkf.second;

        if( pkf == nullptr ) {
            continue;
        }

        vkfids.push_back(kfid);
        vTwc.push_back(map_id_posespar_.at(kfid).getPose());

        for( const auto &kp : pkf->getKeypoints3d() ) {
            int lmid = kp.lmid_;
            if( processed_lmids.count(lmid) ) {
                continue;
            }
            auto plm = pmap_->getMapPoint(lmid);
            if( plm == nullptr ) {
                pmap_->removeMapPointObs(lmid, kfid);
                continue;
            }

            if( plm->kfid_ == kfid ) {
                Eigen::Vector3d campt = pkf->projWorldToCam(plm->getPoint());
                Eigen::Vector3d wpt = vTwc.back() * campt ;

                vlmids.push_back(lmid);
                vwpt.push_back(wpt);

                processed_lmids.insert(lmid);
            }
        }
    }

    std::lock_guard<std::mutex> lock(pmap_->map_mutex_);

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.LC_PoseGraph_Update");

    // Propagate corrections to youngest KFs / MPs
    for( int kfid = newframe.kfid_+1 ; kfid <= pmap_->nkfid_ ; kfid++ ) 
    {
        auto pkf = pmap_->getKeyframe(kfid);
        if( pkf == nullptr ) {
            continue;
        }
        Sophus::SE3d prevTwkf = pkf->getTwc();
        Sophus::SE3d relTlckf_kf = iniTcw * prevTwkf;

        Sophus::SE3d updTwkf = newoptTwc * relTlckf_kf;

        for( const auto &kp : pkf->getKeypoints3d() ) {
            int lmid = kp.lmid_;
            if( processed_lmids.count(lmid) ) {
                continue;
            }
            auto plm = pmap_->getMapPoint(lmid);
            if( plm == nullptr ) {
                pmap_->removeMapPointObs(lmid, kfid);
                continue;
            }
            
            if( plm->kfid_ == kfid ) {
                Eigen::Vector3d campt = pkf->projWorldToCam(plm->getPoint());
                Eigen::Vector3d wpt = updTwkf * campt;

                pmap_->updateMapPoint(lmid, wpt);

                processed_lmids.insert(lmid);
            }
        }

        pkf->setTwc(updTwkf);
    }
    
    // Update KFs / MPs
    for( size_t i = 0, iend = vlmids.size() ; i < iend ; i++ )
    {
        int lmid = vlmids.at(i);
        pmap_->updateMapPoint(lmid, vwpt.at(i));
    }

    for( size_t i = 0, iend = vkfids.size() ; i < iend ; i++ )
    {
        int kfid = vkfids.at(i);
        auto pkf = pmap_->getKeyframe(kfid);
        if( pkf != nullptr ) {
            pkf->setTwc(vTwc.at(i));
        }
    }

    // Update cur frame pose
    Sophus::SE3d prevTwcur = pmap_->pcurframe_->getTwc();
    Sophus::SE3d relTlckf_cur = iniTcw * prevTwcur;

    Sophus::SE3d updTwcur = newoptTwc * relTlckf_cur;

    pmap_->pcurframe_->setTwc(updTwcur);
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.LC_PoseGraph_Update");

    return true;
}


void Optimizer::structureOnlyBA(const std::vector<int> &vlm2optids) 
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.LC_StructBA_setup");

    ceres::Problem problem;
    ceres::LossFunctionWrapper *loss_function;
    
    // Chi2 thresh.
    const float mono_th = pslamstate_->robust_mono_th_;

    loss_function = new ceres::LossFunctionWrapper(new ceres::HuberLoss(std::sqrt(mono_th)), ceres::TAKE_OWNERSHIP);

    auto ordering = new ceres::ParameterBlockOrdering;

    std::unordered_map<int, PoseParametersBlock> map_id_posespar_;
    std::unordered_map<int, PointXYZParametersBlock> map_id_pointspar_;

    // Only going to be used for calib params.
    auto &newframe = *pmap_->getKeyframe(0);

    // Add the left cam calib parameters
    auto pcalibleft = newframe.pcalib_leftcam_;
    CalibParametersBlock calibpar(0, pcalibleft->fx_, pcalibleft->fy_, pcalibleft->cx_, pcalibleft->cy_);
    problem.AddParameterBlock(calibpar.values(), 4);
    ordering->AddElementToGroup(calibpar.values(), 1);

    problem.SetParameterBlockConstant(calibpar.values());

    // Prepare variables if STEREO mode
    auto pcalibright = newframe.pcalib_rightcam_;
    CalibParametersBlock rightcalibpar;
    
    Sophus::SE3d Trl, Tlr;
    PoseParametersBlock rlextrinpose(0, Trl);

    if( pslamstate_->stereo_ ) {
        // Right Intrinsic
        rightcalibpar = CalibParametersBlock(0, pcalibright->fx_, pcalibright->fy_, pcalibright->cx_, pcalibright->cy_);
        problem.AddParameterBlock(rightcalibpar.values(), 4);
        ordering->AddElementToGroup(rightcalibpar.values(), 1);

        problem.SetParameterBlockConstant(rightcalibpar.values());

        // Right Extrinsic
        Tlr = pcalibright->getExtrinsic();
        Trl = Tlr.inverse();
        rlextrinpose = PoseParametersBlock(0, Trl);

        ceres::LocalParameterization *local_param = new SE3LeftParameterization();

        problem.AddParameterBlock(rlextrinpose.values(), 7, local_param);
        ordering->AddElementToGroup(rlextrinpose.values(), 1);

        problem.SetParameterBlockConstant(rlextrinpose.values());
    }

    for( const auto &lmid : vlm2optids ) {

        auto plm = pmap_->getMapPoint(lmid);

        if( plm == nullptr ) {
            continue;
        }

        // Add Map Point
        map_id_pointspar_.emplace(lmid, PointXYZParametersBlock(lmid, plm->getPoint()));

        problem.AddParameterBlock(map_id_pointspar_.at(lmid).values(), 3);
        ordering->AddElementToGroup(map_id_pointspar_.at(lmid).values(), 0);

        // Add KFs as fixed elements.
        for( const auto &kfid : plm->getKfObsSet() ) 
        {
            auto pkf = pmap_->getKeyframe(kfid);

            if( pkf == nullptr ) {
                continue;
            }

            auto kp = pkf->getKeypointById(lmid);

            if( kp.lmid_ != lmid ) {
                continue;
            }

            if( !map_id_posespar_.count(kfid) ) 
            {
                map_id_posespar_.emplace(kfid, PoseParametersBlock(kfid, pkf->getTwc()));

                ceres::LocalParameterization *local_parameterization = new SE3LeftParameterization();

                problem.AddParameterBlock(map_id_posespar_.at(kfid).values(), 7, local_parameterization);
                ordering->AddElementToGroup(map_id_posespar_.at(kfid).values(), 1);

                problem.SetParameterBlockConstant(map_id_posespar_.at(kfid).values());
            }

            ceres::CostFunction *f;

            // Add a visual factor between KF-MP nodes
            if( kp.is_stereo_ ) {
                f = new DirectLeftSE3::ReprojectionErrorKSE3XYZ(
                            kp.unpx_.x, kp.unpx_.y, std::pow(2.,kp.scale_)
                        );

                problem.AddResidualBlock(
                            f, loss_function, 
                            calibpar.values(),
                            map_id_posespar_.at(kfid).values(), 
                            map_id_pointspar_.at(lmid).values()
                        );

                f = new DirectLeftSE3::ReprojectionErrorRightCamKSE3XYZ(
                        kp.runpx_.x, kp.runpx_.y, std::pow(2.,kp.scale_)
                    );

                problem.AddResidualBlock(
                        f, loss_function, 
                        rightcalibpar.values(),
                        map_id_posespar_.at(kfid).values(), 
                        rlextrinpose.values(), 
                        map_id_pointspar_.at(lmid).values()
                    );            
            } 
            else {
                f = new DirectLeftSE3::ReprojectionErrorKSE3XYZ(
                            kp.unpx_.x, kp.unpx_.y, std::pow(2.,kp.scale_)
                        );

                problem.AddResidualBlock(
                            f, loss_function, calibpar.values(), 
                            map_id_posespar_.at(kfid).values(), 
                            map_id_pointspar_.at(lmid).values()
                        );
            }
        }
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.LC_StructBA_setup");

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.LC_StructBA_Optimize");

    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    options.num_threads = 1;

    options.max_num_iterations = 10;
    options.function_tolerance = 1.e-3;
    options.max_solver_time_in_seconds = 0.01;

    if( !pslamstate_->bforce_realtime_ ) {
        options.max_solver_time_in_seconds *= 2.;
    }
    
    options.minimizer_progress_to_stdout = pslamstate_->debug_;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if( pslamstate_->debug_ )
        std::cout << summary.FullReport() << std::endl;

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.LC_StructBA_Optimize");

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.LC_StructBA_Update");

    for( const auto &lmid : vlm2optids ) 
    {
        auto optlmit = map_id_pointspar_.find(lmid);

        if( optlmit != map_id_pointspar_.end() ) {
            pmap_->updateMapPoint(lmid, optlmit->second.getPoint());
        }
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.LC_StructBA_Update");
}


bool Optimizer::fullPoseGraph(std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> &vTwc, 
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> &vTpc, std::vector<bool> &viskf)
{
    ceres::Problem problem;
    
    std::map<int, PoseParametersBlock> map_id_posespar_;

    size_t nbposes = vTwc.size();

    size_t nbkfs = 0;

    for( size_t i = 0 ; i < nbposes ; i++ )
    {
        map_id_posespar_.emplace(i, PoseParametersBlock(i, vTwc.at(i)));

        ceres::LocalParameterization *local_parameterization = new SE3LeftParameterization();
        problem.AddParameterBlock(map_id_posespar_.at(i).values(), 7, local_parameterization);

        if( viskf.at(i) ) {
            problem.SetParameterBlockConstant(map_id_posespar_.at(i).values());
            nbkfs++;
        }

        if( i == 0 ) {
            continue;
        }

        ceres::CostFunction* f
            = new LeftSE3RelativePoseError(vTpc.at(i));
        
        problem.AddResidualBlock(f, nullptr, map_id_posespar_.at(i-1).values(), map_id_posespar_.at(i).values());
    }

    if( pslamstate_->debug_ )
        std::cout << "\n\n - [fullPoseGraph] Going to optimize over " << nbkfs 
            << " KFs / " << nbposes << " poses in total!\n\n";

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    options.max_num_iterations = 100;
    options.function_tolerance = 1.e-6;
    
    options.minimizer_progress_to_stdout = pslamstate_->debug_;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if( pslamstate_->debug_ )
        std::cout << summary.FullReport() << std::endl;
    
    std::ofstream f;
    std::string filename = "ov2slam_full_traj_wlc_opt.txt";

    if( pslamstate_->debug_ )
        std::cout << "\n Going to write the full Pose Graph trajectory into : " 
            << filename << "\n";

    f.open(filename.c_str());
    f << std::fixed;

    for( size_t i = 0 ; i < nbposes ; i++ )
    {
        auto newkfpose = map_id_posespar_.at(i);
        Sophus::SE3d Twc = newkfpose.getPose();

        Eigen::Vector3d twc = Twc.translation();
        Eigen::Quaterniond qwc = Twc.unit_quaternion();

        f << std::setprecision(9) << i << ". " << twc.x() << " " << twc.y() << " " << twc.z()
            << " " << qwc.x() << " " << qwc.y() << " " << qwc.z() << " " << qwc.w() << std::endl;

        f.flush();
    }

    f.close();

    if( pslamstate_->debug_ )
        std::cout << "\nFullPoseGrah Trajectory w. LC file written!\n";

    return true;
}
