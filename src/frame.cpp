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

#include "frame.hpp"

Frame::Frame()
    : id_(-1), kfid_(0), img_time_(0.), nbkps_(0), nb2dkps_(0), nb3dkps_(0), nb_stereo_kps_(0),
      Frl_(Eigen::Matrix3d::Zero()), Fcv_(cv::Mat::zeros(3,3,CV_64F))
{}


Frame::Frame(std::shared_ptr<CameraCalibration> pcalib_left, const size_t ncellsize)
    : id_(-1), kfid_(0), img_time_(0.), ncellsize_(ncellsize), nbkps_(0),
      nb2dkps_(0), nb3dkps_(0), nb_stereo_kps_(0),
      pcalib_leftcam_(pcalib_left)
{
    // Init grid from images size
    nbwcells_ = static_cast<size_t>(ceilf( static_cast<float>(pcalib_leftcam_->img_w_) / ncellsize_ ));
    nbhcells_ = static_cast<size_t>(ceilf( static_cast<float>(pcalib_leftcam_->img_h_) / ncellsize_ ));
    ngridcells_ =  nbwcells_ * nbhcells_ ;
    noccupcells_ = 0;
    
    vgridkps_.resize( ngridcells_ );
}


Frame::Frame(std::shared_ptr<CameraCalibration> pcalib_left, std::shared_ptr<CameraCalibration> pcalib_right, const size_t ncellsize)
    : id_(-1), kfid_(0), img_time_(0.), ncellsize_(ncellsize), nbkps_(0), nb2dkps_(0), nb3dkps_(0), nb_stereo_kps_(0),
    pcalib_leftcam_(pcalib_left), pcalib_rightcam_(pcalib_right)
{
    Eigen::Vector3d t = pcalib_rightcam_->Tcic0_.translation();
    Eigen::Matrix3d tskew;
    tskew << 0., -t(2), t(1),
            t(2), 0., -t(0),
            -t(1), t(0), 0.;

    Eigen::Matrix3d R = pcalib_rightcam_->Tcic0_.rotationMatrix();

    Frl_ = pcalib_rightcam_->K_.transpose().inverse() * tskew * R * pcalib_leftcam_->iK_;

    cv::eigen2cv(Frl_, Fcv_);

    // Init grid from images size
    nbwcells_ = ceil( (float)pcalib_leftcam_->img_w_ / ncellsize_ );
    nbhcells_ = ceil( (float)pcalib_leftcam_->img_h_ / ncellsize_ );
    ngridcells_ =  nbwcells_ * nbhcells_ ;
    noccupcells_ = 0;
    
    vgridkps_.resize( ngridcells_ );
}

Frame::Frame(const Frame &F)
    : id_(F.id_), kfid_(F.kfid_), img_time_(F.img_time_), mapkps_(F.mapkps_), vgridkps_(F.vgridkps_), ngridcells_(F.ngridcells_), noccupcells_(F.noccupcells_),
    ncellsize_(F.ncellsize_), nbwcells_(F.nbwcells_), nbhcells_(F.nbhcells_), nbkps_(F.nbkps_), nb2dkps_(F.nb2dkps_), nb3dkps_(F.nb3dkps_), 
    nb_stereo_kps_(F.nb_stereo_kps_), Twc_(F.Twc_), Tcw_(F.Tcw_), pcalib_leftcam_(F.pcalib_leftcam_),
    pcalib_rightcam_(F.pcalib_rightcam_), Frl_(F.Frl_), Fcv_(F.Fcv_), map_covkfs_(F.map_covkfs_), set_local_mapids_(F.set_local_mapids_)
{}

// Set the image time and id
void Frame::updateFrame(const int id, const double time) 
{
    id_= id;
    img_time_ = time;
}

// Return vector of keypoint objects
std::vector<Keypoint> Frame::getKeypoints() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {
        v.push_back(kp.second);
    }
    return v;
}


// Return vector of 2D keypoint objects
std::vector<Keypoint> Frame::getKeypoints2d() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> v;
    v.reserve(nb2dkps_);
    for( const auto & kp : mapkps_ ) {
        if( !kp.second.is3d_ ) {
            v.push_back(kp.second);
        }
    }
    return v;
}

// Return vector of 3D keypoint objects
std::vector<Keypoint> Frame::getKeypoints3d() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> v;
    v.reserve(nb3dkps_);
    for( const auto &kp : mapkps_ ) {
        if( kp.second.is3d_ ) {
            v.push_back(kp.second);
        }
    }
    return v;
}

// Return vector of stereo keypoint objects
std::vector<Keypoint> Frame::getKeypointsStereo() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> v;
    v.reserve(nb_stereo_kps_);
    for( const auto &kp : mapkps_ ) {
        if( kp.second.is_stereo_ ) {
            v.push_back(kp.second);
        }
    }
    return v;
}

// Return vector of keypoints' raw pixel positions
std::vector<cv::Point2f> Frame::getKeypointsPx() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<cv::Point2f> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {
        v.push_back(kp.second.px_);
    }
    return v;
}

// Return vector of keypoints' undistorted pixel positions
std::vector<cv::Point2f> Frame::getKeypointsUnPx() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<cv::Point2f> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {
        v.push_back(kp.second.unpx_);
    }
    return v;
}

// Return vector of keypoints' bearing vectors
std::vector<Eigen::Vector3d> Frame::getKeypointsBv() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Eigen::Vector3d> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {
        v.push_back(kp.second.bv_);
    }
    return v;
}

// Return vector of keypoints' related landmarks' id
std::vector<int> Frame::getKeypointsId() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<int> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {
        v.push_back(kp.first);
    }
    return v;
}

Keypoint Frame::getKeypointById(const int lmid) const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return Keypoint();
    }

    return it->second;
}


std::vector<Keypoint> Frame::getKeypointsByIds(const std::vector<int> &vlmids) const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> vkp;
    vkp.reserve(vlmids.size());
    for( const auto &lmid : vlmids ) {
        auto it = mapkps_.find(lmid);
        if( it != mapkps_.end() ) {
            vkp.push_back(it->second);
        }
    }

    return vkp;
}


// Return vector of keypoints' descriptor
std::vector<cv::Mat> Frame::getKeypointsDesc() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<cv::Mat> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {
        v.push_back(kp.second.desc_);
    }

    return v;
}


// Compute keypoint from raw pixel position
inline void Frame::computeKeypoint(const cv::Point2f &pt, Keypoint &kp)
{
    kp.px_ = pt;
    kp.unpx_ = pcalib_leftcam_->undistortImagePoint(pt);

    Eigen::Vector3d hunpx(kp.unpx_.x, kp.unpx_.y, 1.);
    kp.bv_ = pcalib_leftcam_->iK_ * hunpx;
    kp.bv_.normalize();
}

// Create keypoint from raw pixel position
inline Keypoint Frame::computeKeypoint(const cv::Point2f &pt, const int lmid)
{
    Keypoint kp;
    kp.lmid_ = lmid;
    computeKeypoint(pt,kp);
    return kp;
}


// Add keypoint object to vector of kps
void Frame::addKeypoint(const Keypoint &kp)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    if( mapkps_.count(kp.lmid_) ) {
        std::cout << "\nWEIRD!  Trying to add a KP with an already existing lmid... Not gonna do it!\n";
        return;
    }

    mapkps_.emplace(kp.lmid_, kp);
    addKeypointToGrid(kp);

    nbkps_++;
    if( kp.is3d_ ) {
        nb3dkps_++;
    } else {
        nb2dkps_++;
    }
}

// Add new keypoint from raw pixel position
void Frame::addKeypoint(const cv::Point2f &pt, const int lmid)
{
    Keypoint kp = computeKeypoint(pt, lmid);

    addKeypoint(kp);
}

// Add new keypoint w. desc
void Frame::addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc) 
{
    Keypoint kp = computeKeypoint(pt, lmid);
    kp.desc_ = desc;

    addKeypoint(kp);
}

// Add new keypoint w. desc & scale
void Frame::addKeypoint(const cv::Point2f &pt, const int lmid, const int scale)
{
    Keypoint kp = computeKeypoint(pt, lmid);
    kp.scale_ = scale;

    addKeypoint(kp);
}

// Add new keypoint w. desc & scale
void Frame::addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc, const int scale)
{
    Keypoint kp = computeKeypoint(pt, lmid);
    kp.desc_ = desc;
    kp.scale_ = scale;

    addKeypoint(kp);
}

// Add new keypoint w. desc & scale & angle
void Frame::addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc, const int scale, const float angle)
{
    Keypoint kp = computeKeypoint(pt, lmid);
    kp.desc_ = desc;
    kp.scale_ = scale;
    kp.angle_ = angle;

    addKeypoint(kp);
}

void Frame::updateKeypoint(const int lmid, const cv::Point2f &pt)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    } 

    Keypoint upkp = it->second;

    if( upkp.is_stereo_ ) {
        nb_stereo_kps_--;
        upkp.is_stereo_ = false;
    }

    computeKeypoint(pt, upkp);
    
    updateKeypointInGrid(it->second, upkp);
    it->second = upkp;
}

void Frame::updateKeypointDesc(const int lmid, const cv::Mat &desc)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }

    it->second.desc_ = desc;
}

void Frame::updateKeypointAngle(const int lmid, const float angle)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }

    it->second.angle_ = angle;
}

bool Frame::updateKeypointId(const int prevlmid, const int newlmid, const bool is3d)
{
    std::unique_lock<std::mutex> lock(kps_mutex_);

    if( mapkps_.count(newlmid) ) {
        return false;
    }

    auto it = mapkps_.find(prevlmid);
    if( it == mapkps_.end() ) {
        return false;
    }

    Keypoint upkp = it->second;
    lock.unlock();
    upkp.lmid_ = newlmid;
    upkp.is_retracked_ = true;
    upkp.is3d_ = is3d;
    removeKeypointById(prevlmid);
    addKeypoint(upkp);

    return true;
}

// Compute stereo keypoint from raw pixel position
void Frame::computeStereoKeypoint(const cv::Point2f &pt, Keypoint &kp)
{
    kp.rpx_ = pt;
    kp.runpx_ = pcalib_rightcam_->undistortImagePoint(pt);

    Eigen::Vector3d bv(kp.runpx_.x, kp.runpx_.y, 1.);
    bv = pcalib_rightcam_->iK_ * bv.eval();
    bv.normalize();

    kp.rbv_ = bv;

    if( !kp.is_stereo_ ) {
        kp.is_stereo_ = true;
        nb_stereo_kps_++;
    }
}


void Frame::updateKeypointStereo(const int lmid, const cv::Point2f &pt)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }

    computeStereoKeypoint(pt, it->second);
}

inline void Frame::removeKeypoint(const Keypoint &kp)
{
    removeKeypointById(kp.lmid_);
}

void Frame::removeKeypointById(const int lmid)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }

    removeKeypointFromGrid(it->second);

    if( it->second.is3d_ ) {
        nb3dkps_--;
    } else {
        nb2dkps_--;
    }
    nbkps_--;
    if( it->second.is_stereo_ ) {
        nb_stereo_kps_--;
    }
    mapkps_.erase(lmid);
}


inline void Frame::removeStereoKeypoint(const Keypoint &kp)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    removeStereoKeypointById(kp.lmid_);
}

void Frame::removeStereoKeypointById(const int lmid)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }
    
    if( it->second.is_stereo_ ) {
        it->second.is_stereo_ = false;
        nb_stereo_kps_--;
    }
}

void Frame::turnKeypoint3d(const int lmid)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }

    if( !it->second.is3d_ ) {
        it->second.is3d_ = true;
        nb3dkps_++;
        nb2dkps_--;
    }
}

bool Frame::isObservingKp(const int lmid) const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);
    return mapkps_.count(lmid);
}

void Frame::addKeypointToGrid(const Keypoint &kp)
{
    std::lock_guard<std::mutex> lock(grid_mutex_);

    int idx = getKeypointCellIdx(kp.px_);

    if( vgridkps_.at(idx).empty() ) {
        noccupcells_++;
    }

    vgridkps_.at(idx).push_back(kp.lmid_);
}

void Frame::removeKeypointFromGrid(const Keypoint &kp)
{
    std::lock_guard<std::mutex> lock(grid_mutex_);

    int idx = getKeypointCellIdx(kp.px_);

    if( idx < 0 || idx >= (int)vgridkps_.size() ) {
        return;
    }

    for( size_t i = 0, iend = vgridkps_.at(idx).size() ; i < iend ; i++ )
    {
        if( vgridkps_.at(idx).at(i) == kp.lmid_ ) {
            vgridkps_.at(idx).erase(vgridkps_.at(idx).begin() + i);

            if( vgridkps_.at(idx).empty() ) {
                noccupcells_--;
            }
            break;
        }
    }
}

void Frame::updateKeypointInGrid(const Keypoint &prevkp, const Keypoint &newkp)
{
    // First ensure that new kp should move
    int idx = getKeypointCellIdx(prevkp.px_);

    int nidx = getKeypointCellIdx(newkp.px_);

    if( idx == nidx ) {
        // Nothing to do
        return;
    }
    else {
        // First remove kp
        removeKeypointFromGrid(prevkp);
        // Second the new kp is added to the grid
        addKeypointToGrid(newkp);
    }
}

std::vector<Keypoint> Frame::getKeypointsFromGrid(const cv::Point2f &pt) const
{
    std::lock_guard<std::mutex> lock(grid_mutex_);

    std::vector<int> voutkpids;

    int idx = getKeypointCellIdx(pt);

    if( idx < 0 || idx >= (int)vgridkps_.size() ) {
        return std::vector<Keypoint>();
    }

    if( vgridkps_.at(idx).empty() ) {
        return std::vector<Keypoint>();
    }

    for( const auto &id : vgridkps_.at(idx) )
    {
        voutkpids.push_back(id);
    }

    return getKeypointsByIds(voutkpids);
}

int Frame::getKeypointCellIdx(const cv::Point2f &pt) const
{
    int r = floor(pt.y / ncellsize_);
    int c = floor(pt.x / ncellsize_);
    return (r * nbwcells_ + c);
}

std::vector<Keypoint> Frame::getSurroundingKeypoints(const Keypoint &kp) const
{
    std::vector<Keypoint> vkps;
    vkps.reserve(20);

    int rkp = floor(kp.px_.y / ncellsize_);
    int ckp = floor(kp.px_.x / ncellsize_);

    std::lock_guard<std::mutex> lock(kps_mutex_);
    std::lock_guard<std::mutex> glock(grid_mutex_);
    
    for( int r = rkp-1 ; r < rkp+1 ; r++ ) {
        for( int c = ckp-1 ; c < ckp+1 ; c++ ) {
            int idx = r * nbwcells_ + c;
            if( r < 0 || c < 0 || idx > (int)vgridkps_.size() ) {
                continue;
            }
            for( const auto &id : vgridkps_.at(idx) ) {
                if( id != kp.lmid_ ) {
                    auto it = mapkps_.find(id);
                    if( it != mapkps_.end() ) {
                        vkps.push_back(it->second);
                    }
                }
            }
        }
    }
    return vkps;
}

std::vector<Keypoint> Frame::getSurroundingKeypoints(const cv::Point2f &pt) const
{
    std::vector<Keypoint> vkps;
    vkps.reserve(20);

    int rkp = floor(pt.y / ncellsize_);
    int ckp = floor(pt.x / ncellsize_);

    std::lock_guard<std::mutex> lock(kps_mutex_);
    std::lock_guard<std::mutex> glock(grid_mutex_);
    
    for( int r = rkp-1 ; r < rkp+1 ; r++ ) {
        for( int c = ckp-1 ; c < ckp+1 ; c++ ) {
            int idx = r * nbwcells_ + c;
            if( r < 0 || c < 0 || idx > (int)vgridkps_.size() ) {
                continue;
            }
            for( const auto &id : vgridkps_.at(idx) ) {
                auto it = mapkps_.find(id);
                if( it != mapkps_.end() ) {
                    vkps.push_back(it->second);
                }
            }
        }
    }
    return vkps;
}

std::map<int,int> Frame::getCovisibleKfMap() const
{
    std::lock_guard<std::mutex> lock(cokfs_mutex_);
    return map_covkfs_;
}

inline void Frame::updateCovisibleKfMap(const std::map<int,int> &cokfs)
{
    std::lock_guard<std::mutex> lock(cokfs_mutex_);
    map_covkfs_ = cokfs;
}

void Frame::addCovisibleKf(const int kfid)
{
    if( kfid == kfid_ ) {
        return;
    }

    std::lock_guard<std::mutex> lock(cokfs_mutex_);
    auto it = map_covkfs_.find(kfid);
    if( it != map_covkfs_.end() ) {
        it->second += 1;
    } else {
        map_covkfs_.emplace(kfid, 1);
    }
}

void Frame::removeCovisibleKf(const int kfid)
{
    if( kfid == kfid_ ) {
        return;
    }

    std::lock_guard<std::mutex> lock(cokfs_mutex_);
    map_covkfs_.erase(kfid);
}

void Frame::decreaseCovisibleKf(const int kfid)
{
    if( kfid == kfid_ ) {
        return;
    }

    std::lock_guard<std::mutex> lock(cokfs_mutex_);
    auto it = map_covkfs_.find(kfid);
    if( it != map_covkfs_.end() ) {
        if( it->second != 0 ) {
            it->second -= 1;
            if( it->second == 0 ) {
                map_covkfs_.erase(it);
            }
        }
    }
}

Sophus::SE3d Frame::getTcw() const
{
    std::lock_guard<std::mutex> lock(pose_mutex_);
    return Tcw_;
}

Sophus::SE3d Frame::getTwc() const
{
    std::lock_guard<std::mutex> lock(pose_mutex_);
    return Twc_;
}

Eigen::Matrix3d Frame::getRcw() const
{
   std::lock_guard<std::mutex> lock(pose_mutex_);
   return Tcw_.rotationMatrix();
}

Eigen::Matrix3d Frame::getRwc() const
{
   std::lock_guard<std::mutex> lock(pose_mutex_);
   return Twc_.rotationMatrix();
}

Eigen::Vector3d Frame::gettcw() const
{
   std::lock_guard<std::mutex> lock(pose_mutex_);
   return Tcw_.translation();
}

Eigen::Vector3d Frame::gettwc() const
{
   std::lock_guard<std::mutex> lock(pose_mutex_);
   return Twc_.translation();
}

void Frame::setTwc(const Sophus::SE3d &Twc)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    Twc_ = Twc;
    Tcw_ = Twc.inverse();
}

inline void Frame::setTcw(const Sophus::SE3d &Tcw)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    Tcw_ = Tcw;
    Twc_ = Tcw.inverse();
}

void Frame::setTwc(const Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    Twc_.setRotationMatrix(Rwc);
    Twc_.translation() = twc;

    Tcw_ = Twc_.inverse();
}


inline void Frame::setTcw(const Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    Tcw_.setRotationMatrix(Rcw);
    Tcw_.translation() = tcw;

    Twc_ = Tcw_.inverse();
}

cv::Point2f Frame::projCamToImage(const Eigen::Vector3d &pt) const
{
    return pcalib_leftcam_->projectCamToImage(pt);
}

cv::Point2f Frame::projCamToRightImage(const Eigen::Vector3d &pt) const
{
    return pcalib_rightcam_->projectCamToImage(pcalib_rightcam_->Tcic0_ * pt);
}

cv::Point2f Frame::projCamToImageDist(const Eigen::Vector3d &pt) const
{
    return pcalib_leftcam_->projectCamToImageDist(pt);
}


cv::Point2f Frame::projCamToRightImageDist(const Eigen::Vector3d &pt) const
{
    return pcalib_rightcam_->projectCamToImageDist(pcalib_rightcam_->Tcic0_ * pt);
}


Eigen::Vector3d Frame::projCamToWorld(const Eigen::Vector3d &pt) const
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    Eigen::Vector3d wpt = Twc_ * pt;

    return wpt;
}

Eigen::Vector3d Frame::projWorldToCam(const Eigen::Vector3d &pt) const
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    Eigen::Vector3d campt = Tcw_ * pt;

    return campt;
}

cv::Point2f Frame::projWorldToImage(const Eigen::Vector3d &pt) const
{
    return pcalib_leftcam_->projectCamToImage(projWorldToCam(pt));
}

cv::Point2f Frame::projWorldToImageDist(const Eigen::Vector3d &pt) const
{
    return pcalib_leftcam_->projectCamToImageDist(projWorldToCam(pt));
}

cv::Point2f Frame::projWorldToRightImage(const Eigen::Vector3d &pt) const
{
    return pcalib_rightcam_->projectCamToImage(pcalib_rightcam_->Tcic0_ * projWorldToCam(pt));
}

cv::Point2f Frame::projWorldToRightImageDist(const Eigen::Vector3d &pt) const
{
    return pcalib_rightcam_->projectCamToImageDist(pcalib_rightcam_->Tcic0_ * projWorldToCam(pt));
}

bool Frame::isInImage(const cv::Point2f &pt) const
{
    return (pt.x >= 0 && pt.y >= 0 && pt.x < pcalib_leftcam_->img_w_ && pt.y < pcalib_leftcam_->img_h_);
}

bool Frame::isInRightImage(const cv::Point2f &pt) const
{
    return (pt.x >= 0 && pt.y >= 0 && pt.x < pcalib_rightcam_->img_w_ && pt.y < pcalib_rightcam_->img_h_);
}

void Frame::displayFrameInfo()
{
    std::cout << "\n************************************";
    std::cout << "\nFrame #" << id_ << " (KF #" << kfid_ << ") info:\n";
    std::cout << "\n> Nb kps all (2d / 3d / stereo) : " << nbkps_ << " (" << nb2dkps_ << " / " << nb3dkps_ << " / " << nb_stereo_kps_ << ")";
    std::cout << "\n> Nb covisible kfs : " << map_covkfs_.size();
    std::cout << "\n twc : " << Twc_.translation().transpose();
    std::cout << "\n************************************\n\n";
}

void Frame::reset()
{
    id_ = -1;
    kfid_ = 0;
    img_time_ = 0.;

    std::lock_guard<std::mutex> lock(kps_mutex_);
    std::lock_guard<std::mutex> lock2(grid_mutex_);
    
    mapkps_.clear();
    vgridkps_.clear();
    vgridkps_.resize( ngridcells_ );

    nbkps_ = 0;
    nb2dkps_ = 0;
    nb3dkps_ = 0;
    nb_stereo_kps_ = 0;

    noccupcells_ = 0;

    Twc_ = Sophus::SE3d();
    Tcw_ = Sophus::SE3d();

    map_covkfs_.clear();
    set_local_mapids_.clear();
}