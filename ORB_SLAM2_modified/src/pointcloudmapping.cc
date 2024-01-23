/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "pointcloudmapping.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/projection_matrix.h>
#include "Converter.h"
#include <pcl/io/pcd_io.h>

#include <boost/make_shared.hpp>

PointCloudMapping::PointCloudMapping(double resolution_)
{
    this->resolution = resolution_;
    voxel.setLeafSize( resolution, resolution, resolution);
    globalMap = boost::make_shared< PointCloud >( );

    viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );
    colorImgs.push_back( color.clone() );
    depthImgs.push_back( depth.clone() );

    keyFrameUpdated.notify_one();
}

pcl::PointCloud<PointTWithID>::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    pcl::PointCloud<PointTWithID>::Ptr tmp(new pcl::PointCloud<PointTWithID>());
    
    // 生成tmp点云
    for ( int m=0; m<depth.rows; m+=3 ){
        for ( int n=0; n<depth.cols; n+=3 )
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d>10)
                continue;
            PointTWithID p;
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;

            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];

            p.kfID = kf->mnId;

            tmp->points.push_back(p);
        }
    }

    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(kf->GetPose());
    pcl::PointCloud<PointTWithID>::Ptr cloud(new pcl::PointCloud<PointTWithID>());
    pcl::transformPointCloud(*tmp, *cloud, T.inverse().matrix());
    
    // 在变换后的点云中设置关键帧ID
    for (auto& point : cloud->points)
    {
        point.kfID = kf->mnId;
    }

    cloud->is_dense = false;

    return cloud;
}


void PointCloudMapping::viewer()
{
    pcl::visualization::PCLVisualizer viewer("viewer");
    viewer.setBackgroundColor(0, 0, 0);
    viewer.addCoordinateSystem(1.0);
    viewer.initCameraParameters();

    while (!viewer.wasStopped())
    {
        {
            unique_lock<mutex> lck_shutdown(shutDownMutex);
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated(keyFrameUpdateMutex);
            keyFrameUpdated.wait(lck_keyframeUpdated);
        }

        // keyframe is updated
        size_t N = 0;
        {
            unique_lock<mutex> lck(keyframeMutex);
            N = keyframes.size();
        }

        for (size_t i = lastKeyframeSize; i < N; i++)
        {
            auto p = generatePointCloud(keyframes[i], colorImgs[i], depthImgs[i]);
            *globalMap += *p;
        }
        pcl::io::savePCDFileBinary("vslam.pcd", *globalMap);
        pcl::PointCloud<PointTWithID>::Ptr tmp(new pcl::PointCloud<PointTWithID>());
        voxel.setInputCloud(globalMap);
        voxel.filter(*tmp);
        globalMap->swap(*tmp);

        // 更新PCLVisualizer中的点云
        viewer.removeAllPointClouds();
        pcl::visualization::PointCloudColorHandlerRGBField<PointTWithID> rgb(globalMap);
        viewer.addPointCloud<PointTWithID>(globalMap, rgb, "sample cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");

        viewer.spinOnce();
        cout << "show global map, size=" << globalMap->points.size() << endl;
        lastKeyframeSize = N;
    }
}

