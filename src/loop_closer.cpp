
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

#include "loop_closer.hpp"

#include <thread>

#include "multi_view_geometry.hpp"

#include <opencv2/features2d.hpp>

#ifdef OPENCV_CONTRIB
  #include <opencv2/xfeatures2d.hpp>
  cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> pbriefd_  = cv::xfeatures2d::BriefDescriptorExtractor::create();
#else
  cv::Ptr<cv::DescriptorExtractor> pbriefd_  = cv::ORB::create(500, 1., 0);
#endif

cv::Ptr<cv::FeatureDetector> pfastd_ = cv::FastFeatureDetector::create(20);


#ifdef IBOW_LCD
LoopCloser::LoopCloser(std::shared_ptr<SlamParams> pslamstate, std::shared_ptr<MapManager> pmap)
    : lcparams_()
    , lcdetetector_(lcparams_), pslamstate_(pslamstate), pmap_(pmap)
    , poptimizer_( new Optimizer(pslamstate_, pmap_) )
{
    std::cout << "\n LoopCloser Object is created!\n";

    vkfids_.reserve(5000);
}
#else
LoopCloser::LoopCloser(std::shared_ptr<SlamParams> pslamstate, std::shared_ptr<MapManager> pmap)
    : pslamstate_(pslamstate), pmap_(pmap)
    , poptimizer_( new Optimizer(pslamstate_, pmap_) )
{
    std::cout << "\n LoopCloser won't be used!\n";
}
#endif

void LoopCloser::run()
{
#ifdef IBOW_LCD
    std::cout << "\n Use LoopCLoser : " << pslamstate_->buse_loop_closer_;
    if( !pslamstate_->buse_loop_closer_ ) {
        return;
    }

    std::cout << "\n LoopCloser is ready to process Keyframes!\n";

    while( !bexit_required_ ) {

        if( getNewKf() ) 
        {
            if( pnewkf_ == nullptr ) {
                continue;
            }

            if( pslamstate_->debug_ || pslamstate_->log_timings_ )
                Profiler::Start("0.LC_ProcessKF");

            // Preprocess KF before using it for LC
            pslamstate_->lckfid_ = pnewkf_->kfid_;

            auto vkps = pnewkf_->getKeypoints();
            size_t nbkps = vkps.size();
            std::vector<cv::KeyPoint> vcvkps;
            cv::Mat cvdescs;
            vcvkps.reserve(nbkps + 300);

            cv::Mat mask = cv::Mat(newkfimg_.rows, newkfimg_.cols, CV_8UC1, cv::Scalar(255));

            for( const auto &kp : vkps ) {
                auto plm = pmap_->getMapPoint(kp.lmid_);
                if( plm == nullptr ) {
                    continue;
                } 
                else if ( !plm->desc_.empty() ) 
                {
                    vcvkps.push_back(cv::KeyPoint(kp.px_, 5., kp.angle_, 1., kp.scale_));
                    cvdescs.push_back(plm->desc_);

                    cv::circle(mask, kp.px_, 2., 0, -1);
                }
            }

            if( cvdescs.empty() ) {
                continue;
            }

            std::vector<cv::KeyPoint> vaddkps;

            // This reduces the probabibilty of having two simultaneous detections
            // in different threads which might slow down the front-end
            // (due to the fact that detection is already performed in parallel)
            {
            std::lock_guard<std::mutex> lock(pmap_->map_mutex_);
            }
            pfastd_->detect(newkfimg_, vaddkps, mask);

            if( !vaddkps.empty() ) {
                cv::KeyPointsFilter::retainBest(vaddkps, 300);
                
                cv::Mat adddescs;
                pbriefd_->compute(newkfimg_, vaddkps, adddescs);

                if( !adddescs.empty() ) {
                    vcvkps.insert(vcvkps.end(), vaddkps.begin(), vaddkps.end());
                    cv::vconcat(cvdescs, adddescs, cvdescs);
                }

                if( pslamstate_->debug_ ) 
                    std::cout << "\n [LoopCloser] >>> Adding KF #" << pnewkf_->kfid_ 
                        << " with (slam + lc) : " << nbkps << " + " << vaddkps.size()
                        << " kps to insert into Vocabulary Tree";
            }

            if( cvdescs.empty() ) {
                continue;
            }

            // Send proc. KF to LC Detector
            kf_idx_++;
            vkfids_.push_back(pnewkf_->kfid_);

            ibow_lcd::LCDetectorResult result;
            lcdetetector_.process(kf_idx_, vcvkps, cvdescs, &result);
            
            if( pslamstate_->debug_ || pslamstate_->log_timings_ )
                Profiler::StopAndDisplay(pslamstate_->debug_, "0.LC_ProcessKF");

            switch (result.status) {
                case ibow_lcd::LC_DETECTED:
                    if( pslamstate_->debug_ )
                        std::cout << "--- Loop detected!!!: " << result.train_id 
                            << " with " << result.inliers << " inliers\n";
                    break;
                default:
                    continue;
            }

            // If LCD returned a candidate, process it
            processLoopCandidate(result.train_id);

            pslamstate_->lckfid_ = -1;

        } else {
            std::chrono::microseconds dura(100);
            std::this_thread::sleep_for(dura);
        }
    }
    std::cout << "\n LoopCloser is stopping!\n";

#else
    return;
#endif
}


void LoopCloser::processLoopCandidate(int kfloopidx)
{
    // Get the KF stored in the vector of processed KF
    int kfid = vkfids_.at(kfloopidx);

    auto plckf = pmap_->getKeyframe(kfid);

    // If not in the map anymore, get the closest one
    while( plckf == nullptr ) {
        kfid--;
        plckf = pmap_->getKeyframe(kfid);
    }

    if( pslamstate_->debug_ )
        std::cout << "\n Testing candidate KF #" << kfid << "(im #" 
            << plckf->id_ << " / cur im #" << pnewkf_->id_ << " )! \n";

    auto cov_map = pnewkf_->getCovisibleKfMap();
    auto it = cov_map.find(kfid);
    if( it != cov_map.end() ) {
        if( it->second > 30 ) {
            if( pslamstate_->debug_ )
                std::cout << "\n KF already covisible ! Skipping!\n";
            return;
        }
    }

    // Pair of matched cur kp / map points
    std::vector<std::pair<int,int>> vkplmids;

    // Do a knnMatching to get a first set of matches
    knnMatching(*pnewkf_, *plckf, vkplmids);

    if( vkplmids.size() < 15 ) {
        return;
    }

    // Do a 2d-2d epipolar based filtering
    std::vector<int> voutliers_idx;
    bool success = epipolarFiltering(*pnewkf_, *plckf, vkplmids, voutliers_idx);

    size_t nbinliers = vkplmids.size() - voutliers_idx.size();

    if( !success || nbinliers < 10 ) {
        if( pslamstate_->debug_ )
            std::cout << "\n Not enough inliers for LC after epipolar filtering\n";
        return;
    }

    if( !voutliers_idx.empty() ) {
        // Remove outliers from vector of pairs
        removeOutliers(vkplmids, voutliers_idx);
    }

    // Do a P3P-Ransac            
    Sophus::SE3d Twc = pnewkf_->getTwc();

    if( pslamstate_->debug_ )
        std::cout << "\n Kf loop pos : " << Twc.translation().transpose() << "\n";

    success = p3pRansac(*pnewkf_, vkplmids, voutliers_idx, Twc);

    if( pslamstate_->debug_ )
        std::cout << "\n Kf loop p3p pos : " << Twc.translation().transpose() << "\n";

    nbinliers = vkplmids.size() - voutliers_idx.size();

    if( !success || nbinliers < 5 ) {
        if( pslamstate_->debug_ )
            std::cout << "\n Not enough inliers for LC after p3pRansac\n";
        return;
    }

    if( pslamstate_->debug_ )
        std::cout << "\n Nb 2D / 3D matches : " << vkplmids.size() << "\n";

    if( !voutliers_idx.empty() ) {
        // Remove outliers from vector of pairs
        removeOutliers(vkplmids, voutliers_idx);
    }

    if( pslamstate_->debug_ )
        std::cout << "\n Nb 2D / 3D matches : " << vkplmids.size() << "\n";

    // Search more MPs in LC KF local Map
    trackLoopLocalMap(*pnewkf_, *plckf, Twc, 10., pslamstate_->fmax_desc_dist_*1.5, vkplmids);

    if( pslamstate_->debug_ )
        std::cout << "\n Nb 2D / 3D matches : " << vkplmids.size() << "\n";

    // If additionnal matches found, p3p again
    if( vkplmids.size() > nbinliers )
    {
        success = computePnP(*pnewkf_, vkplmids, Twc, voutliers_idx);

        if( pslamstate_->debug_ )
            std::cout << "\n Kf loop pnp pos : " 
                << Twc.translation().transpose() << "\n";

        nbinliers = vkplmids.size() - voutliers_idx.size();

        if( pslamstate_->debug_ )
            std::cout << "\n Nb 2D / 3D matches : " << nbinliers << "\n";
    
        if( !success || nbinliers < 30 ) {
            if( pslamstate_->debug_ )
                std::cout << "\n Not enough inliers for LC after p3pRansac\n";
            return;
        }

        if( !voutliers_idx.empty() ) {
            // Remove outliers from vector of pairs
            removeOutliers(vkplmids, voutliers_idx);
        }
    } else {
        return;
    }

    size_t nbgoodkps = vkplmids.size();

    // Pose Graph Optim!
    if( nbgoodkps >= 30 ) 
    {
        std::cout << "\n\n [PoseGraph] >>> Closing a loop between : "
            << " KF #" << pnewkf_->kfid_ << " (img #" << pnewkf_->id_ << ") and KF #"
            << plckf->kfid_ << " (img #" << plckf->id_ << " ).";

        // Notify that LC optim is going on
        pslamstate_->blc_is_on_ = true;

        int inikfid = plckf->getCovisibleKfMap().begin()->first;

        std::unique_lock<std::mutex> lock2(pmap_->optim_mutex_);

        double lc_pose_err = (pnewkf_->getTcw() * Twc).log().norm();

        bool goodlc = poptimizer_->localPoseGraph(*pnewkf_, inikfid, Twc);

        if( pslamstate_->debug_ )
            std::cout << "\n Pose Graph Optimized !\n";

        if( goodlc ) {

            if( pslamstate_->debug_ )
                std::cout << "\n Going to merge matches !\n";

            std::vector<int> vmergedlmids;
            vmergedlmids.reserve(vkplmids.size());

            // Lock the map before merging
            std::unique_lock<std::mutex> lock(pmap_->map_mutex_);

            for( const auto &kplmid : vkplmids )
            {
                int prevlmid = kplmid.first;
                int newlmid = kplmid.second;

                if( prevlmid == newlmid ) {
                    continue;
                }
                
                pmap_->mergeMapPoints(prevlmid, newlmid);

                vmergedlmids.push_back(newlmid);
            }

            if( pslamstate_->debug_ )
                std::cout << "\n Merging Done ! \n";

            poptimizer_->structureOnlyBA(vmergedlmids);

            lock.unlock();
            lock2.unlock();

            if( lc_pose_err >= 0.02 ) 
            {    
                bool buse_robust_cost = true;

                auto pinikf = pmap_->getKeyframe(inikfid);
                inikfid = pinikf->getCovisibleKfMap().begin()->first;

                if( pslamstate_->debug_ )
                    std::cout << "\nLooseBA starting! \n";

                poptimizer_->looseBA(inikfid, pnewkf_->kfid_, buse_robust_cost);

                std::cout << "\n [PoseGraph] >>> LooseBA done!";
            }
        }
    
        pslamstate_->blc_is_on_ = false; 
    }
}

void LoopCloser::knnMatching(const Frame &newkf, const Frame &lckf, std::vector<std::pair<int,int>> &vkplmids)
{
    std::vector<int> vkpids, vlmids;
    vkpids.reserve(newkf.nb3dkps_);
    vlmids.reserve(newkf.nb3dkps_);

    std::vector<int> vgoodkpids, vgoodlmids;
    vgoodkpids.reserve(newkf.nb3dkps_);
    vgoodlmids.reserve(newkf.nb3dkps_);

    cv::Mat query;
    cv::Mat train;

    for( const auto &kp : newkf.getKeypoints() ) 
    {
        if( lckf.isObservingKp(kp.lmid_) && kp.is3d_ ) {
            vkplmids.push_back(std::pair<int,int>(kp.lmid_, kp.lmid_));
        } 
        else {
            auto plm = pmap_->getMapPoint(kp.lmid_);
            if( plm == nullptr ) {
                continue;
            } else if ( !plm->desc_.empty() ) {
                query.push_back(plm->desc_);
                vkpids.push_back(kp.lmid_);
            }
        }
    }

    for( const auto &kp : lckf.getKeypoints3d() ) 
    {
        if( newkf.isObservingKp(kp.lmid_) ) {
            continue;
        } else {
            auto plm = pmap_->getMapPoint(kp.lmid_);
            if( plm == nullptr ) {
                continue;
            } else if ( !plm->desc_.empty() ) {
                train.push_back(plm->desc_);
                vlmids.push_back(kp.lmid_);
            }
        }
    }

    if( query.empty() || train.empty() ) {
        return;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch> > vmatches;
    matcher.knnMatch(query, train, vmatches, 2);

    const int maxdist = query.cols * 0.5 * 8.;

    for( const auto &m : vmatches ) 
    {
        bool bgood = false;
        if( m.size() < 2 ) {
            bgood = true;
        }
        else if( m.at(0).distance <= maxdist &&
            m.at(0).distance <= m.at(1).distance * 0.85 ) 
        {
            bgood = true;
        }

        if( bgood ) {
            int kpid = vkpids.at(m.at(0).queryIdx);
            int lmid = vlmids.at(m.at(0).trainIdx);
            vkplmids.push_back(std::pair<int,int>(kpid, lmid));
        }
    }

    if( vkplmids.empty() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n No matches found for LC! Skipping\n ";
        return;
    }

    if( pslamstate_->debug_ )
        std::cout << "\n Found #" << vkplmids.size() << " matches between loop KFs!\n";
}


bool LoopCloser::epipolarFiltering(const Frame &newkf, const Frame &lckf, std::vector<std::pair<int,int>> &vkplmids, std::vector<int> &voutliers_idx)
{
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    size_t nbkps = vkplmids.size();

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vlcbvs, vcurbvs;
    vlcbvs.reserve(nbkps);
    vcurbvs.reserve(nbkps);
    voutliers_idx.reserve(nbkps);

    for( const auto &kplmid : vkplmids ) {
        auto kp = newkf.getKeypointById(kplmid.first);
        vcurbvs.push_back(kp.bv_);

        auto lckp = lckf.getKeypointById(kplmid.second);
        vlcbvs.push_back(lckp.bv_);
    }

    bool success = 
        MultiViewGeometry::compute5ptEssentialMatrix(
            vlcbvs, vcurbvs, 10 * pslamstate_->nransac_iter_, 
            pslamstate_->fransac_err_, 
            false, pslamstate_->bdo_random, 
            newkf.pcalib_leftcam_->fx_, 
            newkf.pcalib_leftcam_->fy_, 
            R, t, 
            voutliers_idx
            );


    if( pslamstate_->debug_ )
        std::cout << "\n 2D-2D Epipolar outliers #" << voutliers_idx.size() 
            << " / tot #" << vkplmids.size() << " matches between loop KFs!\n";

    return success;
}


void LoopCloser::trackLoopLocalMap(const Frame &newkf, const Frame &lckf, const Sophus::SE3d &Twc, const float maxdist, const float ratio, std::vector<std::pair<int,int>> &vkplmids)
{
    // Search LC local map for more matches
    std::unordered_set<int> set_local_lmids, set_checked_kpids;

    auto lccov_map = lckf.getCovisibleKfMap();
    // Make sure that we will add the LC KF 3d kps
    lccov_map[lckf.kfid_] = 100;

    if( pslamstate_->debug_ )
        std::cout << "\n Co LC KF #" << lckf.kfid_ << " : ";

    for( const auto &cokf : lccov_map )
    {
        int kfid = cokf.first;

        if( kfid < lckf.kfid_ - 15 ) {
            continue;
        } else if( kfid > lckf.kfid_ + 15 ) {
            break;
        }

        auto pcokf = pmap_->getKeyframe(kfid);
        if( pcokf == nullptr ) {
            continue;
        }

        if( pslamstate_->debug_ )
            std::cout << kfid << ", ";

        // Go through their 3d kps
        for( const auto &kp : pcokf->getKeypoints3d() ) {

            auto it = set_checked_kpids.find( kp.lmid_ );

            if( it == set_checked_kpids.end() ) 
            {
                set_checked_kpids.insert(kp.lmid_);

                // If observed and not in the vector of pairs, add it
                // Else, add it to the set of local MPs to track
                if( newkf.isObservingKp(kp.lmid_) ) {
                    std::pair<int,int> kplmid(kp.lmid_, kp.lmid_);
                    auto kpit = std::find(vkplmids.begin(), vkplmids.end(), kplmid);
                    if( kpit == vkplmids.end() ) {
                        vkplmids.push_back(kplmid);
                    }
                } else {
                    set_local_lmids.insert(kp.lmid_);
                }
            }
        }
    }

    std::vector<int> vmatchedkpids;
    vmatchedkpids.reserve(vkplmids.size());

    for( const auto &kplmid : vkplmids ) {
        vmatchedkpids.push_back(kplmid.first);
        set_local_lmids.erase(kplmid.second);
    }

    if( pslamstate_->debug_ )
        std::cout << "\n Searching more matches in local map of size : " 
            << set_local_lmids.size() << " \n";

    Sophus::SE3d Tcw = Twc.inverse();

    std::map<int,int> map_previd_newid = matchToMap(newkf, Tcw, maxdist, ratio, vmatchedkpids, set_local_lmids);

    if( pslamstate_->debug_ )
        std::cout << "\n Match To Map found #" << map_previd_newid.size() 
            << " additionnal matches !\n";

    if( !map_previd_newid.empty() ) 
    {
        for( const auto &kpid_lmid : map_previd_newid ) {
            std::pair<int,int> kplmid(kpid_lmid.first, kpid_lmid.second);
            vkplmids.push_back(kplmid);
        }
    }
}


std::map<int,int> LoopCloser::matchToMap(const Frame &frame, const Sophus::SE3d &Tcw, const float fmaxprojerr, 
    const float fdistratio, const std::vector<int> &vmatchedkpids, std::unordered_set<int> &set_local_lmids)
{
    std::map<int,int> map_previd_newid;
    // Leave if local map is empty
    if( set_local_lmids.empty() ) {
        return map_previd_newid;
    }

    const float vfov = 0.5 * frame.pcalib_leftcam_->img_h_ * frame.pcalib_leftcam_->fy_;
    const float hfov = 0.5 * frame.pcalib_leftcam_->img_w_ * frame.pcalib_leftcam_->fx_;

    float maxradfov = 0.;
    if( hfov > vfov ) {
        maxradfov = std::atan(hfov);
    } else {
        maxradfov = std::atan(hfov);
    }

    const float view_th = std::cos(maxradfov);

    float dmaxpxdist = fmaxprojerr;

    std::map<int, std::vector<std::pair<int, float>>> map_kpids_vlmidsdist;

    // Go through all MP from the local map
    for( const int lmid : set_local_lmids )
    {
        if( frame.isObservingKp(lmid) ) {
            continue;
        }

        auto plm = pmap_->getMapPoint(lmid);

        if( plm == nullptr ) {
            continue;
        } else if( !plm->is3d_ || plm->isBad() ) {
            continue;
        }

        Eigen::Vector3d wpt = plm->getPoint();
        cv::Mat lmdesc = plm->desc_;

        if( lmdesc.empty() ) {
            continue;
        }

        //Project 3D MP into KF's image
        Eigen::Vector3d campt = Tcw * wpt;

        if( campt.z() < 0.1 ) {
            continue;
        }

        float view_angle = campt.z() / campt.norm();

        if( fabs(view_angle) < view_th ) {
            continue;
        }

        cv::Point2f projpx = frame.projCamToImageDist(campt);

        if( !frame.isInImage(projpx) ) {
            continue;
        }

        // Get all the kps around the MP's projection
        auto vnearkps = frame.getSurroundingKeypoints(projpx);

        // Find two best matches
        float mindist = plm->desc_.cols * fdistratio * 8.; // * 8 to get bits size
        int bestid = -1;
        int secid = -1;

        float bestdist = mindist;
        float secdist = mindist;

        // float bestpxdist = mindist;

        std::vector<int> vkpids;
        std::vector<float> vpxdist;
        cv::Mat descs;

        // for( const auto & kpid : set_kpids ) 
        for( const auto &kp : vnearkps )
        {
            auto it = std::find(vmatchedkpids.begin(), vmatchedkpids.end(), kp.lmid_);
            if( it != vmatchedkpids.end() ) {
                continue;
            }

            if( kp.lmid_ < 0 ) {
                continue;
            }

            float pxdist = cv::norm(projpx - kp.px_);

            if( pxdist > dmaxpxdist ) {
                continue;
            }

            // Check that this kp and the MP are indeed
            // candidates for matching (by ensuring that they
            // are never both observed in a given KF)
            auto pkplm = pmap_->getMapPoint(kp.lmid_);
            if( pkplm == nullptr ) {
                continue;
            } else if ( pkplm->desc_.empty() ) {
                continue;
            }

            bool is_candidate = true;
            auto set_plmkfs = plm->getKfObsSet();
            for( const auto &kfid : pkplm->getKfObsSet() ) {
                if( set_plmkfs.count(kfid) ) {
                    is_candidate = false;
                    break;
                }
            }
            if( !is_candidate ) {
                continue;
            }

            float dist = plm->computeMinDescDist(*pkplm);

            if( dist <= bestdist ) {
                secdist = bestdist; // Will stay at mindist 1st time
                secid = bestid; // Will stay at -1 1st time

                bestdist = dist;
                bestid = kp.lmid_;
            }
            else if( dist <= secdist ) {
                secdist = dist;
                secid = kp.lmid_;
            }
        }

        if( bestid != -1 && secid != -1 ) {
            if( 0.9 * secdist < bestdist ) {
                bestid = -1;
            }
        }

        if( bestid < 0 ) {
            continue;
        }

        std::pair<int, float> lmid_dist(lmid, bestdist);
        if( !map_kpids_vlmidsdist.count(bestid) ) {
            std::vector<std::pair<int, float>> v(1,lmid_dist);
            map_kpids_vlmidsdist.emplace(bestid, v);
        } else {
            map_kpids_vlmidsdist.at(bestid).push_back(lmid_dist);
        }
    }

    for( const auto &kpid_vlmidsdist : map_kpids_vlmidsdist )
    {
        int kpid = kpid_vlmidsdist.first;

        float bestdist = 1024;
        int bestlmid = -1;

        for( const auto &lmid_dist : kpid_vlmidsdist.second ) {
            if( lmid_dist.second <= bestdist ) {
                bestdist = lmid_dist.second;
                bestlmid = lmid_dist.first;
            }
        }

        if( bestlmid >= 0 ) {
            map_previd_newid.emplace(kpid, bestlmid);
        }
    }

    return map_previd_newid;
}

bool LoopCloser::p3pRansac(const Frame &newkf, std::vector<std::pair<int,int>> &vkplmids, std::vector<int> &voutliers_idx, Sophus::SE3d &Twc)
{
    if( vkplmids.size() < 4 ) {
        if( pslamstate_->debug_ )
            std::cout << "\n Not enough inliers for p3p\n!";
        return false;
    }

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vbvs, vwpts;

    std::vector<int> vkpids;

    size_t nbkps = vkplmids.size();

    vbvs.reserve(nbkps);
    vwpts.reserve(nbkps);
    voutliers_idx.reserve(nbkps);

    std::vector<int> vbadidx;

    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        int kpid = vkplmids.at(i).first;
        int lmid = vkplmids.at(i).second;

        auto plm = pmap_->getMapPoint(lmid);
        if( plm == nullptr ) {
            vbadidx.push_back(i);
            continue;
        }

        auto kp = newkf.getKeypointById(kpid);

        vwpts.push_back(plm->getPoint());
        vbvs.push_back(kp.bv_);
    }

    int k = 0;
    for( const auto &badidx : vbadidx ) {
        vkplmids.erase(vkplmids.begin() + badidx-k);
        k++;
    }

    if( vbvs.size() < 4 ) {
        if( pslamstate_->debug_ )
            std::cout << "\n Not enough pts for p3p\n!";
        return false;
    }

    bool do_optimize = true;

    bool success = 
            MultiViewGeometry::p3pRansac(
                vbvs, vwpts, 10 * pslamstate_->nransac_iter_, 
                pslamstate_->fransac_err_, 
                do_optimize, pslamstate_->bdo_random, 
                newkf.pcalib_leftcam_->fx_, 
                newkf.pcalib_leftcam_->fy_, 
                Twc, voutliers_idx
                );
    
    if( pslamstate_->debug_ )
        std::cout << "\n P3P RANSAC FOR LC : " << voutliers_idx.size() 
            << " outliers / tot " << nbkps << " 2D / 3D matches\n";

    return success;
}


bool LoopCloser::computePnP(const Frame &frame, const std::vector<std::pair<int,int>> &vkplmids, 
    Sophus::SE3d &Twc, std::vector<int> &voutlier_idx)
{
    // Init vector for PnP
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vwpts;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > vkps;
    std::vector<int> vgoodkpidx, vscales, voutidx;

    size_t nbkps = vkplmids.size();

    vgoodkpidx.reserve(nbkps);
    vkps.reserve(nbkps);
    vwpts.reserve(nbkps);
    vscales.reserve(nbkps);
    voutidx.reserve(nbkps);

    // Get kps & MPs
    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        int kpid = vkplmids.at(i).first;
        int lmid = vkplmids.at(i).second;

        auto plm = pmap_->getMapPoint(lmid);
        if( plm == nullptr ) {
            continue;
        } 
        auto kp = frame.getKeypointById(kpid);
        if( kp.lmid_ < 0 ) {
            continue;
        }

        vgoodkpidx.push_back(i);

        vscales.push_back(kp.scale_);
        vwpts.push_back(plm->getPoint());
        
        vkps.push_back(Eigen::Vector2d(kp.unpx_.x, kp.unpx_.y));
    }

    // If at least 3 correspondances, go
    if( vkps.size() >= 3 ) {
        if( pslamstate_->debug_ )
            std::cout << "\n Nb kps used for Ceres PnP : " << vkps.size();

        bool buse_robust = true;
        bool bapply_l2_after_robust = false;

        bool success =
                MultiViewGeometry::ceresPnP(
                                vkps, vwpts, vscales,
                                Twc,
                                10, pslamstate_->robust_mono_th_, buse_robust, bapply_l2_after_robust,
                                frame.pcalib_leftcam_->fx_, frame.pcalib_leftcam_->fy_,
                                frame.pcalib_leftcam_->cx_, frame.pcalib_leftcam_->cy_, voutidx);

        for( const auto &idx : voutidx ) {
            voutlier_idx.push_back(vgoodkpidx.at(idx));
        }

        return success;
    }

    return false;
}

void LoopCloser::removeOutliers(std::vector<std::pair<int,int>> &vkplmids, std::vector<int> &voutliers_idx)
{
    if( voutliers_idx.empty() ) {
        return;
    }

    size_t nbkps = vkplmids.size();
    std::vector<std::pair<int,int>> vkplmidstmp;

    vkplmidstmp.reserve(nbkps);
    
    size_t j = 0;
    for( size_t i = 0 ;  i < nbkps ; i++ ) 
    {
        if( (int)i != voutliers_idx.at(j) ) {
            vkplmidstmp.push_back(vkplmids.at(i));
        } 
        else {
            j++;
            if( j == voutliers_idx.size() ) {
                j = 0;
                voutliers_idx.at(0) = -1;
            }
        }
    }

    vkplmids.swap(vkplmidstmp);

    voutliers_idx.clear();
}


bool LoopCloser::getNewKf()
{
    std::lock_guard<std::mutex> lock(qkf_mutex_);

    // Check if new KF is available
    if( qpkfs_.empty() ) {
        return false;
    } 
    
    // Get most recent KF
    while( qpkfs_.size() > 1 ) {
        qpkfs_.pop();
    }

    pnewkf_ = qpkfs_.front().first;
    newkfimg_ = qpkfs_.front().second;
    qpkfs_.pop();

    return true;
}

void LoopCloser::addNewKf(const std::shared_ptr<Frame> &pkf, const cv::Mat &im)
{
#ifndef IBOW_LCD
    return;
#else
    std::lock_guard<std::mutex> lock(qkf_mutex_);
    qpkfs_.push(std::pair<std::shared_ptr<Frame>, cv::Mat>(pkf, im));
#endif
}
