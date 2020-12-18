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


#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <ceres/ceres.h>

/*
    SE(3) Parametrization such as:
    1. T + dT = Exp(dT) * T 
    2. T o X = T^(-1) * X (i.e. T: cam -> world)  
*/
class SE3LeftParameterization : public ceres::LocalParameterization {
public:
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const 
    {
        Eigen::Map<const Eigen::Vector3d> t(x);
        Eigen::Map<const Eigen::Quaterniond> q(x+3);

        Eigen::Map<const Eigen::Matrix<double,6,1>> vdelta(delta);

        // Left update
        Sophus::SE3d upT = Sophus::SE3d::exp(vdelta) * Sophus::SE3d(q,t);

        Eigen::Map<Eigen::Vector3d> upt(x_plus_delta);
        Eigen::Map<Eigen::Quaterniond> upq(x_plus_delta+3);

        upt = upT.translation();
        upq = upT.unit_quaternion();

        return true;
    }

    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const
    {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > J(jacobian);
        J.topRows<6>().setIdentity();
        J.bottomRows<1>().setZero();
        return true;
    }

    virtual int GlobalSize() const { return 7; }
    virtual int LocalSize() const { return 6; }
};


class LeftSE3RelativePoseError : public ceres::SizedCostFunction<6, 7, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LeftSE3RelativePoseError(const Sophus::SE3d &Tc0c1,
                            const double sigma = 1.)
        : Tc0c1_(Tc0c1)
    {
        sqrt_cov_ = sigma * Eigen::Matrix<double,6,6>::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix<double,6,6> sqrt_cov_, sqrt_info_;
private:
    Sophus::SE3d Tc0c1_;
};


// Cost functions with SE(3) pose parametrized as
// T cam -> world
namespace DirectLeftSE3 {

class ReprojectionErrorKSE3XYZ : public ceres::SizedCostFunction<2, 4, 7, 3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionErrorKSE3XYZ(const double u, const double v,
                            const double sigma = 1.)
        : unpx_(u,v)
    {
        sqrt_cov_ = sigma * Eigen::Matrix2d::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix2d sqrt_cov_, sqrt_info_;
private:
    Eigen::Vector2d unpx_;
};

class ReprojectionErrorRightCamKSE3XYZ : public ceres::SizedCostFunction<2, 4, 7, 7, 3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionErrorRightCamKSE3XYZ(const double u, const double v,
                            const double sigma = 1.)
        : unpx_(u,v)
    {
        sqrt_cov_ = sigma * Eigen::Matrix2d::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix2d sqrt_cov_, sqrt_info_;
private:
    Eigen::Vector2d unpx_;
};


class ReprojectionErrorSE3 : public ceres::SizedCostFunction<2, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionErrorSE3(const double u, const double v,
            double fx, double fy, double cx, double cy, 
            const Eigen::Vector3d &wpt, const double sigma = 1.)
        : unpx_(u,v), wpt_(wpt), fx_(fx), fy_(fy), cx_(cx), cy_(cy)
    {
        sqrt_cov_ = sigma * Eigen::Matrix2d::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix2d sqrt_cov_, sqrt_info_;
private:
    Eigen::Vector2d unpx_;
    Eigen::Vector3d wpt_;
    double fx_, fy_, cx_, cy_;
};


class ReprojectionErrorKSE3AnchInvDepth : public ceres::SizedCostFunction<2, 4, 7, 7, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionErrorKSE3AnchInvDepth(
                            const double u, const double v,
                            const double uanch, const double vanch,
                            const double sigma = 1.)
        : unpx_(u,v), anchpx_(uanch,vanch,1.)
    {
        sqrt_cov_ = sigma * Eigen::Matrix2d::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix2d sqrt_cov_, sqrt_info_;
private:
    Eigen::Vector2d unpx_;
    Eigen::Vector3d anchpx_;
};


class ReprojectionErrorRightAnchCamKSE3AnchInvDepth : public ceres::SizedCostFunction<2, 4, 4, 7, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionErrorRightAnchCamKSE3AnchInvDepth(
                    const double ur, const double vr,
                    const double uanch, const double vanch,
                    const double  sigmar = 1.)
        : runpx_(ur,vr), anchpx_(uanch,vanch,1.)
    {
        sqrt_cov_.setZero();
        sqrt_cov_ = sigmar * Eigen::Matrix2d::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated
    // in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix2d sqrt_cov_, sqrt_info_;
private:
    Eigen::Vector2d runpx_;
    Eigen::Vector3d anchpx_;
};


class ReprojectionErrorRightCamKSE3AnchInvDepth : public ceres::SizedCostFunction<2, 4, 4, 7, 7, 7, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionErrorRightCamKSE3AnchInvDepth(
                    const double ur, const double vr,
                    const double uanch, const double vanch,
                    const double  sigmar = 1.)
        : runpx_(ur,vr), anchpx_(uanch,vanch,1.)
    {
        sqrt_cov_.setZero();
        sqrt_cov_ = sigmar * Eigen::Matrix2d::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated
    // in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix2d sqrt_cov_, sqrt_info_;
private:
    Eigen::Vector2d runpx_;
    Eigen::Vector3d anchpx_;
};


} // end namespace