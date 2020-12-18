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

#include <thread>

#include "estimator.hpp"


void Estimator::run()
{
    std::cout << "\n Estimator is ready to process Keyframes!\n";
    
    while( !bexit_required_ ) {

        if( getNewKf() ) 
        {
            if( pslamstate_->slam_mode_ ) 
            {
                if( pslamstate_->debug_ )
                    std::cout << "\n [Estimator] Slam-Mode - Processing new KF #" << pnewkf_->kfid_;

                applyLocalBA();

                mapFiltering();

            } else {
                if( pslamstate_->debug_ )
                    std::cout << "\nNO OPITMIZATION (NEITHER SLAM MODE NOR SW MODE SELECTED) !\n";
            }
        } else {
            std::chrono::microseconds dura(20);
            std::this_thread::sleep_for(dura);
        }
    }

    poptimizer_->signalStopLocalBA();
    
    std::lock_guard<std::mutex> lock2(pmap_->optim_mutex_);

    std::cout << "\n Estimator thread is exiting.\n";
}


void Estimator::applyLocalBA()
{
    int nmincstkfs = 1;
    if( pslamstate_->mono_ ) {
        nmincstkfs = 2;
    }

    if( pnewkf_->kfid_ < nmincstkfs ) {
        return;
    }

    if( pnewkf_->nb3dkps_ == 0 ) {
        return;
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.BA_localBA");

    std::lock_guard<std::mutex> lock2(pmap_->optim_mutex_);

    // We signal that Estimator is performing BA
    pslamstate_->blocalba_is_on_ = true;

    bool use_robust_cost = true;
    poptimizer_->localBA(*pnewkf_, use_robust_cost);

    // We signal that Estimator is stopping BA
    pslamstate_->blocalba_is_on_ = false;
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.BA_localBA");
}


void Estimator::mapFiltering()
{
    if( pslamstate_->fkf_filtering_ratio_ >= 1. ) {
        return;
    }
    
    if( pnewkf_->kfid_ < 20 || pslamstate_->blc_is_on_ ) {
        return;
    }   

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.BA_map-filtering");
        
    auto covkf_map = pnewkf_->getCovisibleKfMap();

    for( auto it = covkf_map.rbegin() ; it != covkf_map.rend() ; it++ ) {

        int kfid = it->first;

        if( bnewkfavailable_ || kfid == 0 ) {
            break;
        }

        if( kfid >= pnewkf_->kfid_ ) {
            continue;
        }

        // Only useful if LC enabled
        if( pslamstate_->lckfid_ == kfid ) {
            continue;
        }

        auto pkf = pmap_->getKeyframe(kfid);
        if( pkf == nullptr ) {
            pnewkf_->removeCovisibleKf(kfid);
            continue;
        } 
        else if( (int)pkf->nb3dkps_ < pslamstate_->nmin_covscore_ / 2 ) {
            std::lock_guard<std::mutex> lock(pmap_->map_mutex_);
            pmap_->removeKeyframe(kfid);
            continue;
        }

        size_t nbgoodobs = 0;
        size_t nbtot = 0;
        for( const auto &kp : pkf->getKeypoints3d() )
        {
            auto plm = pmap_->getMapPoint(kp.lmid_);
            if( plm == nullptr ) {
                pmap_->removeMapPointObs(kp.lmid_, kfid);
                continue;
            } 
            else if( plm->isBad() ) {
                continue;
            }
            else {
                size_t nbcokfs = plm->getKfObsSet().size();
                if( nbcokfs > 4 ) {
                    nbgoodobs++;
                }
            } 
            
            nbtot++;
            
            if( bnewkfavailable_ ) {
                break;
            }
        }
        float ratio = (float)nbgoodobs / nbtot;
        if( ratio > pslamstate_->fkf_filtering_ratio_ ) {

            // Only useful if LC enabled
            if( pslamstate_->lckfid_ == kfid ) {
                continue;
            }
            std::lock_guard<std::mutex> lock(pmap_->map_mutex_);
            pmap_->removeKeyframe(kfid);
        }
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.BA_map-filtering");
}

bool Estimator::getNewKf()
{
    std::lock_guard<std::mutex> lock(qkf_mutex_);

    // Check if new KF is available
    if( qpkfs_.empty() ) {
        bnewkfavailable_ = false;
        return false;
    }

    // In SLAM-mode, we only processed the last received KF
    // but we trick the covscore if several KFs were waiting
    // to make sure that they are all optimized
    std::vector<int> vkfids;
    vkfids.reserve(qpkfs_.size());
    while( qpkfs_.size() > 1 ) {
        qpkfs_.pop();
        vkfids.push_back(pnewkf_->kfid_);
    }
    pnewkf_ = qpkfs_.front();
    qpkfs_.pop();

    if( !vkfids.empty() ) {
        for( const auto &kfid : vkfids ) {
            pnewkf_->map_covkfs_[kfid] = pnewkf_->nb3dkps_;
        }

        if( pslamstate_->debug_ )
            std::cout << "\n ESTIMATOR is late!  Adding several KFs to BA...\n";
    }
    bnewkfavailable_ = false;

    return true;
}


void Estimator::addNewKf(const std::shared_ptr<Frame> &pkf)
{
    std::lock_guard<std::mutex> lock(qkf_mutex_);
    qpkfs_.push(pkf);
    bnewkfavailable_ = true;

    // We signal that a new KF is ready
    if( pslamstate_->blocalba_is_on_ 
        && !poptimizer_->stopLocalBA() ) 
    {
        poptimizer_->signalStopLocalBA();
    }
}

void Estimator::reset()
{
    std::lock_guard<std::mutex> lock2(pmap_->optim_mutex_);

    bnewkfavailable_ = false;
    bexit_required_ = false; 

    std::queue<std::shared_ptr<Frame>> empty;
    std::swap(qpkfs_, empty);
}