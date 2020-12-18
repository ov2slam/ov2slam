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

#include <opencv2/video/tracking.hpp>

#include "visual_front_end.hpp"
#include "multi_view_geometry.hpp"

#include <opencv2/highgui.hpp>


VisualFrontEnd::VisualFrontEnd(std::shared_ptr<SlamParams> pstate, std::shared_ptr<Frame> pframe, 
        std::shared_ptr<MapManager> pmap, std::shared_ptr<FeatureTracker> ptracker)
    : pslamstate_(pstate), pcurframe_(pframe), pmap_(pmap), ptracker_(ptracker)
{}

bool VisualFrontEnd::visualTracking(cv::Mat &iml, double time)
{
    std::lock_guard<std::mutex> lock(pmap_->map_mutex_);
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("0.Full-Front_End");

    bool iskfreq = trackMono(iml, time);

    if( iskfreq ) {
        pmap_->createKeyframe(cur_img_, iml);

        if( pslamstate_->btrack_keyframetoframe_ ) {
            cv::buildOpticalFlowPyramid(cur_img_, kf_pyr_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
        }
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "0.Full-Front_End");

    return iskfreq;
}


// Perform tracking in one image, update kps and MP obs, return true if a new KF is req.
bool VisualFrontEnd::trackMono(cv::Mat &im, double time)
{
    if( pslamstate_->debug_ )
        std::cout << "\n\n - [Visual-Front-End]: Track Mono Image\n";
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.FE_Track-Mono");

    // Preprocess the new image
    preprocessImage(im);

    // Create KF if 1st frame processed
    if( pcurframe_->id_ == 0 ) {
        return true;
    }
    
    // Apply Motion model to predict cur Frame pose
    Sophus::SE3d Twc = pcurframe_->getTwc();
    motion_model_.applyMotionModel(Twc, time);
    pcurframe_->setTwc(Twc);
    
    // Track the new image
    if( pslamstate_->btrack_keyframetoframe_ ) {
        kltTrackingFromKF();
    } else {
        kltTracking();
    }

    if( pslamstate_->doepipolar_ ) {
        // Check2d2dOutliers
        epipolar2d2dFiltering();
    }

    if( pslamstate_->mono_ && !pslamstate_->bvision_init_ ) 
    {
        if( pcurframe_->nb2dkps_ < 50 ) {
            pslamstate_->breset_req_ = true;
            return false;
        } 
        else if( checkReadyForInit() ) {
            std::cout << "\n\n - [Visual-Front-End]: Mono Visual SLAM ready for initialization!";
            pslamstate_->bvision_init_ = true;
            return true;
        } 
        else {
            std::cout << "\n\n - [Visual-Front-End]: Not ready to init yet!";
            return false;
        }
    }

    // Compute Pose (2D-3D)
    computePose();

    // Update Motion model from estimated pose
    motion_model_.updateMotionModel(pcurframe_->Twc_, time);

    // Check if New KF req.
    bool is_kf_req = checkNewKfReq();

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.FE_Track-Mono");

    return is_kf_req;
}


// KLT Tracking with motion prior
void VisualFrontEnd::kltTracking()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_KLT-Tracking");

    // Get current kps and init priors for tracking
    std::vector<int> v3dkpids, vkpids;
    std::vector<cv::Point2f> v3dkps, v3dpriors, vkps, vpriors;
    std::vector<bool> vkpis3d;

    // First we're gonna track 3d kps on only 2 levels
    v3dkpids.reserve(pcurframe_->nb3dkps_);
    v3dkps.reserve(pcurframe_->nb3dkps_);
    v3dpriors.reserve(pcurframe_->nb3dkps_);

    // Then we'll track 2d kps on full pyramid levels
    vkpids.reserve(pcurframe_->nbkps_);
    vkps.reserve(pcurframe_->nbkps_);
    vpriors.reserve(pcurframe_->nbkps_);

    vkpis3d.reserve(pcurframe_->nbkps_);


    // Front-End is thread-safe so we can direclty access curframe's kps
    for( const auto &it : pcurframe_->mapkps_ ) 
    {
        auto &kp = it.second;

        // Init prior px pos. from motion model
        if( pslamstate_->klt_use_prior_ )
        {
            if( kp.is3d_ ) 
            {
                cv::Point2f projpx = pcurframe_->projWorldToImageDist(pmap_->map_plms_.at(kp.lmid_)->getPoint());

                // Add prior if projected into image
                if( pcurframe_->isInImage(projpx) ) 
                {
                    v3dkps.push_back(kp.px_);
                    v3dpriors.push_back(projpx);
                    v3dkpids.push_back(kp.lmid_);

                    vkpis3d.push_back(true);
                    continue;
                }
            }
        }

        // For other kps init prior with prev px pos.
        vkpids.push_back(kp.lmid_);
        vkps.push_back(kp.px_);
        vpriors.push_back(kp.px_);
    }

    // 1st track 3d kps if using prior
    if( pslamstate_->klt_use_prior_ && !v3dpriors.empty() ) 
    {
        int nbpyrlvl = 1;

        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        auto vprior = v3dpriors;

        ptracker_->fbKltTracking(
                    prev_pyr_, 
                    cur_pyr_, 
                    pslamstate_->nklt_win_size_, 
                    nbpyrlvl, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    v3dkps, 
                    v3dpriors, 
                    vkpstatus);

        size_t nbgood = 0;
        size_t nbkps = v3dkps.size();

        for(size_t i = 0 ; i < nbkps  ; i++ ) 
        {
            if( vkpstatus.at(i) ) {
                pcurframe_->updateKeypoint(v3dkpids.at(i), v3dpriors.at(i));
                nbgood++;
            } else {
                // If tracking failed, gonna try on full pyramid size
                vkpids.push_back(v3dkpids.at(i));
                vkps.push_back(v3dkps.at(i));
                vpriors.push_back(v3dpriors.at(i));
            }
        }

        if( pslamstate_->debug_ ) {
            std::cout << "\n >>> KLT Tracking w. priors : " << nbgood;
            std::cout << " out of " << nbkps << " kps tracked!\n";
        }

        if( nbgood < 0.33 * nbkps ) {
            // Motion model might be quite wrong, P3P is recommended next
            // and not using any prior
            bp3preq_ = true;
            vpriors = vkps;
        }
    }

    // 2nd track other kps if any
    if( !vkps.empty() ) 
    {
        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        ptracker_->fbKltTracking(
                    prev_pyr_, 
                    cur_pyr_, 
                    pslamstate_->nklt_win_size_, 
                    pslamstate_->nklt_pyr_lvl_, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    vkps, 
                    vpriors, 
                    vkpstatus);
        
        size_t nbgood = 0;
        size_t nbkps = vkps.size();

        for(size_t i = 0 ; i < nbkps  ; i++ ) 
        {
            if( vkpstatus.at(i) ) {
                pcurframe_->updateKeypoint(vkpids.at(i), vpriors.at(i));
                nbgood++;
            } else {
                // MapManager is responsible for all the removing operations
                pmap_->removeObsFromCurFrameById(vkpids.at(i));
            }
        }

        if( pslamstate_->debug_ ) {
            std::cout << "\n >>> KLT Tracking no prior : " << nbgood;
            std::cout << " out of " << nbkps << " kps tracked!\n";
        }
    } 
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_KLT-Tracking");
}


void VisualFrontEnd::kltTrackingFromKF()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_KLT-Tracking-from-KF");

    // Get current kps and init priors for tracking
    std::vector<int> v3dkpids, vkpids;
    std::vector<cv::Point2f> v3dkps, v3dpriors, vkps, vpriors;
    std::vector<bool> vkpis3d;

    // First we're gonna track 3d kps on only 2 levels
    v3dkpids.reserve(pcurframe_->nb3dkps_);
    v3dkps.reserve(pcurframe_->nb3dkps_);
    v3dpriors.reserve(pcurframe_->nb3dkps_);

    // Then we'll track 2d kps on full pyramid levels
    vkpids.reserve(pcurframe_->nbkps_);
    vkps.reserve(pcurframe_->nbkps_);
    vpriors.reserve(pcurframe_->nbkps_);

    vkpis3d.reserve(pcurframe_->nbkps_);

    // Get prev KF
    auto pkf = pmap_->map_pkfs_.at(pcurframe_->kfid_);

    if( pkf == nullptr ) {
        return;
    }

    std::vector<int> vbadids;
    vbadids.reserve(pcurframe_->nbkps_ * 0.2);


    // Front-End is thread-safe so we can direclty access curframe's kps
    for( const auto &it : pcurframe_->mapkps_ ) 
    {
        auto &kp = it.second;

        auto kfkpit = pkf->mapkps_.find(kp.lmid_);
        if( kfkpit == pkf->mapkps_.end() ) {
            vbadids.push_back(kp.lmid_);
            continue;
        }

        // Init prior px pos. from motion model
        if( pslamstate_->klt_use_prior_ )
        {
            if( kp.is3d_ ) 
            {
                cv::Point2f projpx = pcurframe_->projWorldToImageDist(pmap_->map_plms_.at(kp.lmid_)->getPoint());

                // Add prior if projected into image
                if( pcurframe_->isInImage(projpx) ) 
                {
                    v3dkps.push_back(kfkpit->second.px_);
                    v3dpriors.push_back(projpx);
                    v3dkpids.push_back(kp.lmid_);

                    vkpis3d.push_back(true);
                    continue;
                }
            }
        }

        // For other kps init prior with prev px pos.
        vkpids.push_back(kp.lmid_);
        vkps.push_back(kfkpit->second.px_);
        vpriors.push_back(kp.px_);
    }

    for( const auto &badid : vbadids ) {
        // MapManager is responsible for all the removing operations
        pmap_->removeObsFromCurFrameById(badid);
    }

    // 1st track 3d kps if using prior
    if( pslamstate_->klt_use_prior_ && !v3dpriors.empty() ) 
    {
        int nbpyrlvl = 1;

        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        auto vprior = v3dpriors;

        ptracker_->fbKltTracking(
                    kf_pyr_, 
                    cur_pyr_, 
                    pslamstate_->nklt_win_size_, 
                    nbpyrlvl, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    v3dkps, 
                    v3dpriors, 
                    vkpstatus);

        size_t nbgood = 0;
        size_t nbkps = v3dkps.size();

        for(size_t i = 0 ; i < nbkps  ; i++ ) 
        {
            if( vkpstatus.at(i) ) {
                pcurframe_->updateKeypoint(v3dkpids.at(i), v3dpriors.at(i));
                nbgood++;
            } else {
                // If tracking failed, gonna try on full pyramid size
                vkpids.push_back(v3dkpids.at(i));
                vkps.push_back(v3dkps.at(i));
                vpriors.push_back(pcurframe_->mapkps_.at(v3dkpids.at(i)).px_);
            }
        }

        if( pslamstate_->debug_ ) {
            std::cout << "\n >>> KLT Tracking w. priors : " << nbgood;
            std::cout << " out of " << nbkps << " kps tracked!\n";
        }

        if( nbgood < 0.33 * nbkps ) {
            // Motion model might be quite wrong, P3P is recommended next
            // and not using any prior
            bp3preq_ = true;
            vpriors = vkps;
        }
    }

    // 2nd track other kps if any
    if( !vkps.empty() ) 
    {
        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        ptracker_->fbKltTracking(
                    kf_pyr_, 
                    cur_pyr_, 
                    pslamstate_->nklt_win_size_, 
                    pslamstate_->nklt_pyr_lvl_, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    vkps, 
                    vpriors, 
                    vkpstatus);
        
        size_t nbgood = 0;
        size_t nbkps = vkps.size();

        for(size_t i = 0 ; i < nbkps  ; i++ ) 
        {
            if( vkpstatus.at(i) ) {
                pcurframe_->updateKeypoint(vkpids.at(i), vpriors.at(i));
                nbgood++;
            } else {
                // MapManager is responsible for all the removing operations
                pmap_->removeObsFromCurFrameById(vkpids.at(i));
            }
        }

        if( pslamstate_->debug_ ) {
            std::cout << "\n >>> KLT Tracking no prior : " << nbgood;
            std::cout << " out of " << nbkps << " kps tracked!\n";
        }
    } 
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_KLT-Tracking");
}


// This function apply a 2d-2d based outliers filtering
void VisualFrontEnd::epipolar2d2dFiltering()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_EpipolarFiltering");
    
    // Get prev. KF (direct access as Front-End is thread safe)
    auto pkf = pmap_->map_pkfs_.at(pcurframe_->kfid_);

    if( pkf == nullptr ) {
        std::cerr << "\nERROR! Previous Kf does not exist yet (epipolar2d2d()).\n";
        exit(-1);
    }

    // Get cur. Frame nb kps
    size_t nbkps = pcurframe_->nbkps_;

    if( nbkps < 8 ) {
        if( pslamstate_->debug_ )
            std::cout << "\nNot enough kps to compute Essential Matrix\n";
        return;
    }

    // Setup Essential Matrix computation for OpenGV-based filtering
    std::vector<int> vkpsids, voutliersidx;
    vkpsids.reserve(nbkps);
    voutliersidx.reserve(nbkps);

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vkfbvs, vcurbvs;
    vkfbvs.reserve(nbkps);
    vcurbvs.reserve(nbkps);
    
    size_t nbparallax = 0;
    float avg_parallax = 0.;

    // In stereo mode, we consider 3d kps as better tracks and therefore
    // use only them for computing E with RANSAC, 2d kps are then removed based
    // on the resulting Fundamental Mat.
    bool epifrom3dkps = false;
    if( pslamstate_->stereo_ && pcurframe_->nb3dkps_ > 30 ) {
        epifrom3dkps = true;
    }

    // Compute rotation compensated parallax
    Eigen::Matrix3d Rkfcur = pkf->getRcw() * pcurframe_->getRwc();

    // Init bearing vectors and check parallax
    for( const auto &it : pcurframe_->mapkps_ ) {

        if( epifrom3dkps ) {
            if( !it.second.is3d_ ) {
                continue;
            }
        }

        auto &kp = it.second;

        // Get the prev. KF related kp if it exists
        auto kfkp = pkf->getKeypointById(kp.lmid_);

        if( kfkp.lmid_ != kp.lmid_ ) {
            continue;
        }

        // Store the bvs and their ids
        vkfbvs.push_back(kfkp.bv_);
        vcurbvs.push_back(kp.bv_);
        vkpsids.push_back(kp.lmid_);

        cv::Point2f rotpx = pkf->projCamToImage(Rkfcur * kp.bv_);

        // Compute parallax
        avg_parallax += cv::norm(rotpx - kfkp.unpx_);
        nbparallax++;
    }

    if( nbkps < 8 ) {
        if( pslamstate_->debug_ )
            std::cout << "\nNot enough kps to compute Essential Matrix\n";
        return;
    }

    // Average parallax
    avg_parallax /= nbparallax;

    if( avg_parallax < 2. * pslamstate_->fransac_err_ ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> Not enough parallax (" << avg_parallax 
                << " px) to compute 5-pt Essential Matrix\n";
        return;
    }

    bool do_optimize = false;

    // In monocular case, we'll use the resulting motion if tracking is poor
    if( pslamstate_->mono_ && pmap_->nbkfs_ > 2 
        && pcurframe_->nb3dkps_ < 30 ) 
    {
        do_optimize = true;
    }

    Eigen::Matrix3d Rkfc;
    Eigen::Vector3d tkfc;

    if( pslamstate_->debug_ ) {
        std::cout << "\n \t>>> 5-pt EssentialMatrix Ransac :";
        std::cout << "\n \t>>> only on 3d kps : " << epifrom3dkps;
        std::cout << "\n \t>>> nb pts : " << nbkps;
        std::cout << " / avg. parallax : " << avg_parallax;
        std::cout << " / nransac_iter_ : " << pslamstate_->nransac_iter_;
        std::cout << " / fransac_err_ : " << pslamstate_->fransac_err_;
        std::cout << "\n\n";
    }
    
    bool success = 
        MultiViewGeometry::compute5ptEssentialMatrix(
                    vkfbvs, vcurbvs, 
                    pslamstate_->nransac_iter_, 
                    pslamstate_->fransac_err_, 
                    do_optimize, 
                    pslamstate_->bdo_random, 
                    pcurframe_->pcalib_leftcam_->fx_, 
                    pcurframe_->pcalib_leftcam_->fy_, 
                    Rkfc, tkfc, 
                    voutliersidx);

    if( pslamstate_->debug_ )
        std::cout << "\n \t>>> Epipolar nb outliers : " << voutliersidx.size();

    if( !success) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> No pose could be computed from 5-pt EssentialMatrix\n";
        return;
    }

    if( voutliersidx.size() > 0.5 * vkfbvs.size() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> Too many outliers, skipping as might be degenerate case\n";
        return;
    }

    // Remove outliers
    for( const auto & idx : voutliersidx ) {
        // MapManager is responsible for all the removing operations.
        pmap_->removeObsFromCurFrameById(vkpsids.at(idx));
    }

    // In case we wanted to use the resulting motion 
    // (mono mode - can help when tracking is poor)
    if( do_optimize && pmap_->nbkfs_ > 2 ) 
    {
        // Get motion model translation scale from last KF
        Sophus::SE3d Tkfw = pkf->getTcw();
        Sophus::SE3d Tkfcur = Tkfw * pcurframe_->getTwc();

        double scale = Tkfcur.translation().norm();
        tkfc.normalize();

        // Update current pose with Essential Mat. relative motion
        // and current trans. scale
        Sophus::SE3d Tkfc(Rkfc, scale * tkfc);

        pcurframe_->setTwc(pkf->getTwc() * Tkfc);
    }

    // In case we only used 3d kps for computing E (stereo mode)
    if( epifrom3dkps ) {

        if( pslamstate_->debug_ )
            std::cout << "\n Applying found Essential Mat to 2D kps!\n";

        Sophus::SE3d Tidentity;
        Sophus::SE3d Tkfcur(Rkfc, tkfc);

        Eigen::Matrix3d Fkfcur = MultiViewGeometry::computeFundamentalMat12(Tidentity, Tkfcur, pcurframe_->pcalib_leftcam_->K_);

        std::vector<int> vbadkpids;
        vbadkpids.reserve(pcurframe_->nb2dkps_);

        for( const auto &it : pcurframe_->mapkps_ ) 
        {
            if( it.second.is3d_ ) {
                continue;
            }

            auto &kp = it.second;

            // Get the prev. KF related kp if it exists
            auto kfkp = pkf->getKeypointById(kp.lmid_);

            // Normalized coord.
            Eigen::Vector3d curpt(kp.unpx_.x, kp.unpx_.y, 1.);
            Eigen::Vector3d kfpt(kfkp.unpx_.x, kfkp.unpx_.y, 1.);

            float epi_err = MultiViewGeometry::computeSampsonDistance(Fkfcur, curpt, kfpt);

            if( epi_err > pslamstate_->fransac_err_ ) {
                vbadkpids.push_back(kp.lmid_);
            }
        }

        for( const auto & kpid : vbadkpids ) {
            pmap_->removeObsFromCurFrameById(kpid);
        }

        if( pslamstate_->debug_ )
            std::cout << "\n Nb of 2d kps removed : " << vbadkpids.size() << " \n";
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_EpipolarFiltering");
}


void VisualFrontEnd::computePose()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_computePose");

    // Get cur nb of 3D kps    
    size_t nb3dkps = pcurframe_->nb3dkps_;

    if( nb3dkps < 4 ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> Not enough kps to compute P3P / PnP\n";
        return;
    }

    // Setup P3P-Ransac computation for OpenGV-based Pose estimation
    // + motion-only BA with Ceres
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vbvs, vwpts;
    std::vector<int> vkpids, voutliersidx, vscales;

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > vkps;

    vbvs.reserve(nb3dkps);
    vwpts.reserve(nb3dkps);
    vkpids.reserve(nb3dkps);
    voutliersidx.reserve(nb3dkps);

    vkps.reserve(nb3dkps);
    vscales.reserve(nb3dkps);

    bool bdop3p = bp3preq_ || pslamstate_->dop3p_;

    // Store every 3D bvs, MPs and their related ids
    for( const auto &it : pcurframe_->mapkps_ ) 
    {
        if( !it.second.is3d_ ) {
            continue;
        }

        auto &kp = it.second;
        // auto plm = pmap_->getMapPoint(kp.lmid_);
        auto plm = pmap_->map_plms_.at(kp.lmid_);
        if( plm == nullptr ) {
            continue;
        }

        if( bdop3p ) {
            vbvs.push_back(kp.bv_);
        }

        vkps.push_back(Eigen::Vector2d(kp.unpx_.x, kp.unpx_.y));
        vwpts.push_back(plm->getPoint());
        vscales.push_back(kp.scale_);
        vkpids.push_back(kp.lmid_);
    }

    Sophus::SE3d Twc = pcurframe_->getTwc();
    bool do_optimize = false;
    bool success = false;

    if( bdop3p ) 
    {
        if( pslamstate_->debug_ ) {
            std::cout << "\n \t>>>P3P Ransac : ";
            std::cout << "\n \t>>> nb 3d pts : " << nb3dkps;
            std::cout << " / nransac_iter_ : " << pslamstate_->nransac_iter_;
            std::cout << " / fransac_err_ : " << pslamstate_->fransac_err_;
            std::cout << "\n\n";
        }

        // Only effective with OpenGV
        bool use_lmeds = true;

        success = 
            MultiViewGeometry::p3pRansac(
                            vbvs, vwpts, 
                            pslamstate_->nransac_iter_, 
                            pslamstate_->fransac_err_, 
                            do_optimize, 
                            pslamstate_->bdo_random, 
                            pcurframe_->pcalib_leftcam_->fx_, 
                            pcurframe_->pcalib_leftcam_->fy_, 
                            Twc,
                            voutliersidx,
                            use_lmeds);

        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> P3P-LMeds nb outliers : " << voutliersidx.size();

        // Check that pose estim. was good enough
        size_t nbinliers = vwpts.size() - voutliersidx.size();

        if( !success
            || nbinliers < 5
            || Twc.translation().array().isInf().any()
            || Twc.translation().array().isNaN().any() )
        {
            if( pslamstate_->debug_ )
                std::cout << "\n \t>>> Not enough inliers for reliable pose est. Resetting KF state\n";

            resetFrame();

            return;
        } 

        // Pose seems to be OK!

        // Update frame pose
        pcurframe_->setTwc(Twc);

        // Remove outliers before PnP refinement (a bit dirty)
        int k = 0;
        for( const auto &idx : voutliersidx ) {
            // MapManager is responsible for all removing operations
            pmap_->removeObsFromCurFrameById(vkpids.at(idx-k));
            vkps.erase(vkps.begin() + idx - k);
            vwpts.erase(vwpts.begin() + idx - k);
            vkpids.erase(vkpids.begin() + idx - k);
            vscales.erase(vscales.begin() + idx - k);
            k++;
        }

        // Clear before robust PnP refinement using Ceres
        voutliersidx.clear();
    }

    // Ceres-based PnP (motion-only BA)
    bool buse_robust = true;
    bool bapply_l2_after_robust = pslamstate_->apply_l2_after_robust_;
    
    size_t nbmaxiters = 5;

    success =
        MultiViewGeometry::ceresPnP(
                        vkps, vwpts, 
                        vscales,
                        Twc, 
                        nbmaxiters, 
                        pslamstate_->robust_mono_th_, 
                        buse_robust, 
                        bapply_l2_after_robust,
                        pcurframe_->pcalib_leftcam_->fx_, pcurframe_->pcalib_leftcam_->fy_,
                        pcurframe_->pcalib_leftcam_->cx_, pcurframe_->pcalib_leftcam_->cy_,
                        voutliersidx);
    
    // Check that pose estim. was good enough
    size_t nbinliers = vwpts.size() - voutliersidx.size();

    if( pslamstate_->debug_ )
        std::cout << "\n \t>>> Ceres PnP nb outliers : " << voutliersidx.size();

    if( !success
        || nbinliers < 5
        || voutliersidx.size() > 0.5 * vwpts.size()
        || Twc.translation().array().isInf().any()
        || Twc.translation().array().isNaN().any() )
    {
        if( !bdop3p ) {
            // Weird results, skipping here and applying p3p next
            bp3preq_ = true;
        }
        else if( pslamstate_->mono_ ) {

            if( pslamstate_->debug_ )
                std::cout << "\n \t>>> Not enough inliers for reliable pose est. Resetting KF state\n";

            resetFrame();
        } 
        // else {
            // resetFrame();
            // motion_model_.reset();
        // }

        return;
    } 

    // Pose seems to be OK!

    // Update frame pose
    pcurframe_->setTwc(Twc);

    // Set p3p req to false as it is triggered either because
    // of bad PnP or by bad klt tracking
    bp3preq_ = false;

    // Remove outliers
    for( const auto & idx : voutliersidx ) {
        // MapManager is responsible for all removing operations
        pmap_->removeObsFromCurFrameById(vkpids.at(idx));
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_computePose");
}



bool VisualFrontEnd::checkReadyForInit()
{
    double avg_rot_parallax = computeParallax(pcurframe_->kfid_, false);

    std::cout << "\n \t>>> Init current parallax (" << avg_rot_parallax <<" px)\n"; 

    if( avg_rot_parallax > pslamstate_->finit_parallax_ ) {
        auto cb = std::chrono::high_resolution_clock::now();
        
        // Get prev. KF
        auto pkf = pmap_->map_pkfs_.at(pcurframe_->kfid_);
        if( pkf == nullptr ) {
            return false;
        }

        // Get cur. Frame nb kps
        size_t nbkps = pcurframe_->nbkps_;

        if( nbkps < 8 ) {
            std::cout << "\nNot enough kps to compute 5-pt Essential Matrix\n";
            return false;
        }

        // Setup Essential Matrix computation for OpenGV-based filtering
        std::vector<int> vkpsids, voutliersidx;
        vkpsids.reserve(nbkps);
        voutliersidx.reserve(nbkps);

        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vkfbvs, vcurbvs;
        vkfbvs.reserve(nbkps);
        vcurbvs.reserve(nbkps);

        Eigen::Matrix3d Rkfcur = pkf->getTcw().rotationMatrix() * pcurframe_->getTwc().rotationMatrix();
        int nbparallax = 0;
        float avg_rot_parallax = 0.;

        // Get bvs and compute the rotation compensated parallax for all cur kps
        // for( const auto &kp : pcurframe_->getKeypoints() ) {
        for( const auto &it : pcurframe_->mapkps_ ) {
            auto &kp = it.second;
            // Get the prev. KF related kp if it exists
            auto kfkp = pkf->getKeypointById(kp.lmid_);

            if( kfkp.lmid_ != kp.lmid_ ) {
                continue;
            }

            // Store the bvs and their ids
            vkfbvs.push_back(kfkp.bv_);
            vcurbvs.push_back(kp.bv_);
            vkpsids.push_back(kp.lmid_);

            // Compute rotation compensated parallax
            Eigen::Vector3d rotbv = Rkfcur * kp.bv_;

            Eigen::Vector3d unpx = pcurframe_->pcalib_leftcam_->K_ * rotbv;
            cv::Point2f rotpx(unpx.x() / unpx.z(), unpx.y() / unpx.z());

            avg_rot_parallax += cv::norm(rotpx - kfkp.unpx_);
            nbparallax++;
        }

        if( nbparallax < 8 ) {
            std::cout << "\nNot enough prev KF kps to compute 5-pt Essential Matrix\n";
            return false;
        }

        // Average parallax
        avg_rot_parallax /= (nbparallax);

        if( avg_rot_parallax < pslamstate_->finit_parallax_ ) {
            std::cout << "\n \t>>> Not enough parallax (" << avg_rot_parallax <<" px) to compute 5-pt Essential Matrix\n";
            return false;
        }

        bool do_optimize = true;

        Eigen::Matrix3d Rkfc;
        Eigen::Vector3d tkfc;
        Rkfc.setIdentity();
        tkfc.setZero();

        std::cout << "\n \t>>> 5-pt EssentialMatrix Ransac :";
        std::cout << "\n \t>>> nb pts : " << nbkps;
        std::cout << " / avg. parallax : " << avg_rot_parallax;
        std::cout << " / nransac_iter_ : " << pslamstate_->nransac_iter_;
        std::cout << " / fransac_err_ : " << pslamstate_->fransac_err_;
        std::cout << " / bdo_random : " << pslamstate_->bdo_random;
        std::cout << "\n\n";
        
        bool success = 
            MultiViewGeometry::compute5ptEssentialMatrix
                    (vkfbvs, vcurbvs, pslamstate_->nransac_iter_, pslamstate_->fransac_err_, 
                    do_optimize, pslamstate_->bdo_random, 
                    pcurframe_->pcalib_leftcam_->fx_, 
                    pcurframe_->pcalib_leftcam_->fy_, 
                    Rkfc, tkfc, 
                    voutliersidx);

        std::cout << "\n \t>>> Epipolar nb outliers : " << voutliersidx.size();

        if( !success ) {
            std::cout << "\n \t>>> No pose could be computed from 5-pt EssentialMatrix\n";
            return false;
        }

        // Remove outliers from cur. Frame
        for( const auto & idx : voutliersidx ) {
            // MapManager is responsible for all the removing operations.
            pmap_->removeObsFromCurFrameById(vkpsids.at(idx));
        }

        // Arbitrary scale
        tkfc.normalize();
        tkfc = tkfc.eval() * 0.25;

        std::cout << "\n \t>>> Essential Mat init : " << tkfc.transpose();

        pcurframe_->setTwc(Rkfc, tkfc);
        
        auto ce = std::chrono::high_resolution_clock::now();
        std::cout << "\n \t>>> Essential Mat Intialization run time : " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(ce-cb).count()
            << "[ms]" << std::endl;

        return true;
    }

    return false;
}

bool VisualFrontEnd::checkNewKfReq()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_checkNewKfReq");

    // Get prev. KF
    auto pkfit = pmap_->map_pkfs_.find(pcurframe_->kfid_);

    if( pkfit == pmap_->map_pkfs_.end() ) {
        return false; // Should not happen
    }
    auto pkf = pkfit->second;

    // Compute median parallax
    double med_rot_parallax = 0.;

    // unrot : false / median : true / only_2d : false
    med_rot_parallax = computeParallax(pkf->kfid_, true, true, false);

    // Id diff with last KF
    int nbimfromkf = pcurframe_->id_-pkf->id_;

    if( pcurframe_->noccupcells_ < 0.33 * pslamstate_->nbmaxkps_
        && nbimfromkf >= 5
        && !pslamstate_->blocalba_is_on_ )
    {
        return true;
    }

    if( pcurframe_->nb3dkps_ < 20 &&
        nbimfromkf >= 2 )
    {
        return true;
    }

    if( pcurframe_->nb3dkps_ > 0.5 * pslamstate_->nbmaxkps_ 
        && (pslamstate_->blocalba_is_on_ || nbimfromkf < 2) )
    {
        return false;
    }

    // Time diff since last KF in sec.
    double time_diff = pcurframe_->img_time_ - pkf->img_time_;

    if( pslamstate_->stereo_ && time_diff > 1. 
        && !pslamstate_->blocalba_is_on_ )
    {
        return true;
    }

    bool cx = med_rot_parallax >= pslamstate_->finit_parallax_ / 2.
        || (pslamstate_->stereo_ && !pslamstate_->blocalba_is_on_ && pcurframe_->id_-pkf->id_ > 2);

    bool c0 = med_rot_parallax >= pslamstate_->finit_parallax_;
    bool c1 = pcurframe_->nb3dkps_ < 0.75 * pkf->nb3dkps_;
    bool c2 = pcurframe_->noccupcells_ < 0.5 * pslamstate_->nbmaxkps_
                && pcurframe_->nb3dkps_ < 0.85 * pkf->nb3dkps_
                && !pslamstate_->blocalba_is_on_;
    
    bool bkfreq = (c0 || c1 || c2) && cx;

    if( bkfreq && pslamstate_->debug_ ) {
        
        std::cout << "\n\n----------------------------------------------------------------------";
        std::cout << "\n>>> Check Keyframe conditions :";
        std::cout << "\n> pcurframe_->id_ = " << pcurframe_->id_ << " / prev kf frame_id : " << pkf->id_;
        std::cout << "\n> Prev KF nb 3d kps = " << pkf->nb3dkps_ << " / Cur Frame = " << pcurframe_->nb3dkps_;
        std::cout << " / Cur Frame occup cells = " << pcurframe_->noccupcells_ << " / parallax = " << med_rot_parallax;
        std::cout << "\n-------------------------------------------------------------------\n\n";
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_checkNewKfReq");

    return bkfreq;
}


// This function computes the parallax (in px.) between cur. Frame 
// and the provided KF id.
float VisualFrontEnd::computeParallax(const int kfid, bool do_unrot, bool bmedian, bool b2donly)
{
    // Get prev. KF
    auto pkfit = pmap_->map_pkfs_.find(kfid);
    
    if( pkfit == pmap_->map_pkfs_.end() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n[Visual Front End] Error in computeParallax ! Prev KF #" 
                    << kfid << " does not exist!\n";
        return 0.;
    }

    // Compute relative rotation between cur Frame 
    // and prev. KF if required
    Eigen::Matrix3d Rkfcur(Eigen::Matrix3d::Identity());
    if( do_unrot ) {
        Eigen::Matrix3d Rkfw = pkfit->second->getRcw();
        Eigen::Matrix3d Rwcur = pcurframe_->getRwc();
        Rkfcur = Rkfw * Rwcur;
    }

    // Compute parallax 
    float avg_parallax = 0.;
    int nbparallax = 0;

    std::set<float> set_parallax;

    // Compute parallax for all kps seen in prev. KF{
    for( const auto &it : pcurframe_->mapkps_ ) 
    {
        if( b2donly && it.second.is3d_ ) {
            continue;
        }

        auto &kp = it.second;
        // Get prev. KF kp if it exists
        auto kfkp = pkfit->second->getKeypointById(kp.lmid_);

        if( kfkp.lmid_ != kp.lmid_ ) {
            continue;
        }

        // Compute parallax with unpx pos.
        cv::Point2f unpx = kp.unpx_;

        // Rotate bv into KF cam frame and back project into image
        if( do_unrot ) {
            unpx = pkfit->second->projCamToImage(Rkfcur * kp.bv_);
        }

        // Compute rotation-compensated parallax
        float parallax = cv::norm(unpx - kfkp.unpx_);
        avg_parallax += parallax;
        nbparallax++;

        if( bmedian ) {
            set_parallax.insert(parallax);
        }
    }

    if( nbparallax == 0 ) {
        return 0.;
    }

    // Average parallax
    avg_parallax /= nbparallax;

    if( bmedian ) 
    {
        auto it = set_parallax.begin();
        std::advance(it, set_parallax.size() / 2);
        avg_parallax = *it;
    }

    return avg_parallax;
}

void VisualFrontEnd::preprocessImage(cv::Mat &img_raw)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_preprocessImage");

    // Set cur raw img
    // left_raw_img_ = img_raw;

    // Update prev img
    if( !pslamstate_->btrack_keyframetoframe_ ) {
        // cur_img_.copyTo(prev_img_);
        cv::swap(cur_img_, prev_img_);
    }

    // Update cur img
    if( pslamstate_->use_clahe_ ) {
        ptracker_->pclahe_->apply(img_raw, cur_img_);
    } else {
        cur_img_ = img_raw;
    }

    // Pre-building the pyramid used for KLT speed-up
    if( pslamstate_->do_klt_ ) {

        // If tracking from prev image, swap the pyramid
        if( !cur_pyr_.empty() && !pslamstate_->btrack_keyframetoframe_ ) {
            prev_pyr_.swap(cur_pyr_);
        }

        cv::buildOpticalFlowPyramid(cur_img_, cur_pyr_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_preprocessImage");
}


// Reset current Frame state
void VisualFrontEnd::resetFrame()
{
    auto mapkps = pcurframe_->mapkps_;
    for( const auto &kpit : mapkps ) {
        pmap_->removeObsFromCurFrameById(kpit.first);
    }
    pcurframe_->mapkps_.clear();
    pcurframe_->vgridkps_.clear();
    pcurframe_->vgridkps_.resize( pcurframe_->ngridcells_ );

    // Do not clear those as we keep the same pose
    // and hence keep a chance to retrack the previous map
    //
    // pcurframe_->map_covkfs_.clear();
    // pcurframe_->set_local_mapids_.clear();

    pcurframe_->nbkps_ = 0;
    pcurframe_->nb2dkps_ = 0;
    pcurframe_->nb3dkps_ = 0;
    pcurframe_->nb_stereo_kps_ = 0;

    pcurframe_->noccupcells_ = 0;
}

// Reset VisualFrontEnd
void VisualFrontEnd::reset()
{
    cur_img_.release();
    prev_img_.release();

    // left_raw_img_.release();

    cur_pyr_.clear();
    prev_pyr_.clear();
    kf_pyr_.clear();
}
