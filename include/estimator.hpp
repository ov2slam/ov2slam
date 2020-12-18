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


#include <queue>
#include <deque>

#include "map_manager.hpp"
#include "optimizer.hpp"

class Estimator {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Estimator(std::shared_ptr<SlamParams> pslamstate, std::shared_ptr<MapManager> pmap)
        : pslamstate_(pslamstate), pmap_(pmap)
        , poptimizer_( new Optimizer(pslamstate_, pmap_) )
    {
        std::cout << "\n Estimator Object is created!\n";
    }

    void run();
    void reset();

    void processKeyframe();

    void applyLocalBA();
    void mapFiltering();

    bool getNewKf();
    void addNewKf(const std::shared_ptr<Frame> &pkf);


    std::shared_ptr<SlamParams> pslamstate_;
    std::shared_ptr<MapManager> pmap_;

    std::unique_ptr<Optimizer> poptimizer_;

    std::shared_ptr<Frame> pnewkf_;

    bool bnewkfavailable_ = false;
    bool bexit_required_ = false;

    bool blooseba_on_ = false;

    std::queue<std::shared_ptr<Frame>> qpkfs_;

    std::mutex qkf_mutex_;
};