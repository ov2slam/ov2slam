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

#include "feature_tracker.hpp"

#include <iostream>
#include <unordered_map>
#include <opencv2/video/tracking.hpp>

#include "multi_view_geometry.hpp"

void FeatureTracker::fbKltTracking(const std::vector<cv::Mat> &vprevpyr, const std::vector<cv::Mat> &vcurpyr, 
        int nwinsize, int nbpyrlvl, float ferr, float fmax_fbklt_dist, std::vector<cv::Point2f> &vkps, 
        std::vector<cv::Point2f> &vpriorkps, std::vector<bool> &vkpstatus) const
{
    // std::cout << "\n \t >>> Forward-Backward kltTracking with Pyr of Images and Motion Prior! \n";

    assert(vprevpyr.size() == vcurpyr.size());

    if( vkps.empty() ) {
        // std::cout << "\n \t >>> No kps were provided to kltTracking()!\n";
        return;
    }

    cv::Size klt_win_size(nwinsize, nwinsize);

    if( (int)vprevpyr.size() < 2*(nbpyrlvl+1) ) {
        nbpyrlvl = vprevpyr.size() / 2 - 1;
    }

    // Objects for OpenCV KLT
    size_t nbkps = vkps.size();
    vkpstatus.reserve(nbkps);

    std::vector<uchar> vstatus;
    std::vector<float> verr;
    std::vector<int> vkpsidx;
    vstatus.reserve(nbkps);
    verr.reserve(nbkps);
    vkpsidx.reserve(nbkps);

    // Tracking Forward
    cv::calcOpticalFlowPyrLK(vprevpyr, vcurpyr, vkps, vpriorkps, 
                vstatus, verr, klt_win_size,  nbpyrlvl, klt_convg_crit_, 
                (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS) 
                );

    std::vector<cv::Point2f> vnewkps;
    std::vector<cv::Point2f> vbackkps;
    vnewkps.reserve(nbkps);
    vbackkps.reserve(nbkps);

    size_t nbgood = 0;

    // Init outliers vector & update tracked kps
    for( size_t i = 0 ; i < nbkps ; i++ ) 
    {
        if( !vstatus.at(i) ) {
            vkpstatus.push_back(false);
            continue;
        }

        if( verr.at(i) > ferr ) {
            vkpstatus.push_back(false);
            continue;
        }

        if( !inBorder(vpriorkps.at(i), vcurpyr.at(0)) ) {
            vkpstatus.push_back(false);
            continue;
        }

        vnewkps.push_back(vpriorkps.at(i));
        vbackkps.push_back(vkps.at(i));
        vkpstatus.push_back(true);
        vkpsidx.push_back(i);
        nbgood++;
    }  

    if( vnewkps.empty() ) {
        return;
    }
    
    vstatus.clear();
    verr.clear();

    // std::cout << "\n \t >>> Forward kltTracking : #" << nbgood << " out of #" << nbkps << " \n";

    // Tracking Backward
    cv::calcOpticalFlowPyrLK(vcurpyr, vprevpyr, vnewkps, vbackkps, 
                vstatus, verr, klt_win_size,  0, klt_convg_crit_,
                (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS) 
                );
    
    nbgood = 0;
    for( int i = 0, iend=vnewkps.size() ; i < iend ; i++ )
    {
        int idx = vkpsidx.at(i);

        if( !vstatus.at(i) ) {
            vkpstatus.at(idx) = false;
            continue;
        }

        if( cv::norm(vkps.at(idx) - vbackkps.at(i)) > fmax_fbklt_dist ) {
            vkpstatus.at(idx) = false;
            continue;
        }

        nbgood++;
    }

    // std::cout << "\n \t >>> Backward kltTracking : #" << nbgood << " out of #" << vkpsidx.size() << " \n";
}


void FeatureTracker::getLineMinSAD(const cv::Mat &iml, const cv::Mat &imr, 
    const cv::Point2f &pt,  const int nwinsize, float &xprior, 
    float &l1err, bool bgoleft) const
{
    xprior = -1;

    if( nwinsize % 2 == 0 ) {
        std::cerr << "\ngetLineMinSAD requires an odd window size\n";
        return;
    }

    const float x = pt.x;
    const float y = pt.y;
    int halfwin = nwinsize / 2;

    if( x - halfwin < 0 ) 
        halfwin += (x-halfwin);
    if( x + halfwin >= imr.cols )
        halfwin += (x+halfwin - imr.cols - 1);
    if( y - halfwin < 0 )
        halfwin += (y-halfwin);
    if( y + halfwin >= imr.rows )
        halfwin += (y+halfwin - imr.rows - 1);
    
    if( halfwin <= 0 ) {
        return;
    }

    cv::Size winsize(2 * halfwin + 1, 2 * halfwin + 1);

    int nbwinpx = (winsize.width * winsize.height);

    float minsad = 255.;
    // int minxpx = -1;

    cv::Mat patch, target;

    cv::getRectSubPix(iml, winsize, pt, patch);

    if( bgoleft ) {
        for( float c = x ; c >= halfwin ; c-=1. )
        {
            cv::getRectSubPix(imr, winsize, cv::Point2f(c, y), target);
            l1err = cv::norm(patch, target, cv::NORM_L1);
            l1err /= nbwinpx;

            if( l1err < minsad ) {
                minsad = l1err;
                xprior = c;
            }
        }
    } else {
        for( float c = x ; c < imr.cols - halfwin ; c+=1. )
        {
            cv::getRectSubPix(imr, winsize, cv::Point2f(c, y), target);
            l1err = cv::norm(patch, target, cv::NORM_L1);
            l1err /= nbwinpx;

            if( l1err < minsad ) {
                minsad = l1err;
                xprior = c;
            }
        }
    }

    l1err = minsad;
}



/**
 * \brief Perform a forward-backward calcOpticalFlowPyrLK() tracking with OpenCV.
 *
 * \param[in] pt  Opencv 2D point.
 * \return True if pt is within image borders, False otherwise
 */
bool FeatureTracker::inBorder(const cv::Point2f &pt, const cv::Mat &im) const
{
    const float BORDER_SIZE = 1.;

    return BORDER_SIZE <= pt.x && pt.x < im.cols - BORDER_SIZE && BORDER_SIZE <= pt.y && pt.y < im.rows - BORDER_SIZE;
}
