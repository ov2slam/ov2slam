/**
* This file is part of ibow-lcd.
*
* Copyright (C) 2017 Emilio Garcia-Fidalgo <emilio.garcia@uib.es> (University of the Balearic Islands)
*
* ibow-lcd is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ibow-lcd is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ibow-lcd. If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>

#include <boost/filesystem.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ibow_lcd/lcdetector.h"

void getFilenames(const std::string& directory,
                  std::vector<std::string>* filenames) {
    using namespace boost::filesystem;

    filenames->clear();
    path dir(directory);

    // Retrieving, sorting and filtering filenames.
    std::vector<path> entries;
    copy(directory_iterator(dir), directory_iterator(), back_inserter(entries));
    sort(entries.begin(), entries.end());
    for (auto it = entries.begin(); it != entries.end(); it++) {
        std::string ext = it->extension().c_str();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".png" || ext == ".jpg" ||
            ext == ".ppm" || ext == ".jpeg") {
            filenames->push_back(it->string());
        }
    }
}

void displayImages(const cv::Mat &im1, const cv::Mat &im2) {
  cv::Mat im;
  cv::hconcat(im1,im2,im);

  cv::namedWindow("lcdetected");
  cv::imshow("lcdetected", im);
  cv::waitKey(0);
  cv::destroyAllWindows();
}

int main(int argc, char** argv) {
  // Creating feature detector and descriptor
  cv::Ptr<cv::Feature2D> detector = cv::ORB::create(500);  // Default params

  cv::Ptr<cv::FastFeatureDetector> pfast_ = cv::FastFeatureDetector::create(20);
  cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> pbrief_  = cv::xfeatures2d::BriefDescriptorExtractor::create();

  // Loading image filenames
  std::vector<std::string> filenames;
  getFilenames(argv[1], &filenames);
  unsigned nimages = filenames.size();

  // Creating the loop closure detector object
  ibow_lcd::LCDetectorParams params;  // Assign desired parameters
  ibow_lcd::LCDetector lcdet(params);

  // Processing the sequence of images
  int k = 5;
  for (unsigned i = 0; i < nimages / k; i++) {
    // Processing image i
    std::cout << "--- Processing image " << i*k << std::endl;

    // Loading and describing the image
    cv::Mat img = cv::imread(filenames[i*k]);
    cv::Mat im;
    cv::cvtColor(img, im, cv::COLOR_RGB2GRAY);
    std::vector<cv::KeyPoint> kps;
    // detector->detect(img, kps);
    pfast_->detect(img, kps);
    cv::KeyPointsFilter::retainBest(kps, 1000);
    // std::vector<cv::Point2f> vkps;
    // cv::goodFeaturesToTrack(im, vkps, 1000, 0.001, 15);
    // cv::KeyPoint::convert(vkps, kps);
    cv::Mat dscs;
    // detector->compute(img, kps, dscs);
    pbrief_->compute(img, kps, dscs);

    ibow_lcd::LCDetectorResult result;
    lcdet.process(i, kps, dscs, &result);

    switch (result.status) {
      case ibow_lcd::LC_DETECTED:
        std::cout << "--- Loop detected!!!: " << result.train_id <<
                     " with " << result.inliers << " inliers" << std::endl;
        displayImages(img, cv::imread(filenames[result.train_id*k]));
        break;
      case ibow_lcd::LC_NOT_DETECTED:
        std::cout << "No loop found" << std::endl;
        break;
      case ibow_lcd::LC_NOT_ENOUGH_IMAGES:
        std::cout << "Not enough images to found a loop" << std::endl;
        break;
      case ibow_lcd::LC_NOT_ENOUGH_ISLANDS:
        std::cout << "Not enough islands to found a loop" << std::endl;
        break;
      case ibow_lcd::LC_NOT_ENOUGH_INLIERS:
        std::cout << "Not enough inliers" << std::endl;
        break;
      case ibow_lcd::LC_TRANSITION:
        std::cout << "Transitional loop closure" << std::endl;
        break;
      default:
        std::cout << "No status information" << std::endl;
        break;
    }
  }

  return 0;

}
