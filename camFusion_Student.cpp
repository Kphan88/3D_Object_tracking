
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
                
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
            
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        //cout<<"DEBUG BB Lidar pts: "<<it1->lidarPoints.size()<<endl;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);
    imwrite("Objects3D.png",topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // find all kpts contained by the given BBox
    std::vector<double> matchedKptDist; 
    for( auto match : kptMatches)
    {   
        cv::KeyPoint kpt_prev = kptsPrev[match.queryIdx];
        cv::KeyPoint kpt_curr = kptsCurr[match.trainIdx];
        if (boundingBox.roi.contains(kpt_curr.pt))
        {
            boundingBox.keypoints.push_back(kpt_curr);
            boundingBox.kptMatches.push_back(match);

            matchedKptDist.push_back(cv::norm(kpt_curr.pt - kpt_prev.pt));
        }

    }
    

    // Filter outliers 
    double averageDistance = std::accumulate(matchedKptDist.begin(),matchedKptDist.end(), 0.0)/matchedKptDist.size();
    std::vector<int> to_delete; 
    for (int i = 0; i < matchedKptDist.size(); ++i)
    {
        if (matchedKptDist[i] > averageDistance)
        {
            to_delete.push_back(i);
        }
    }
    //cout<<"debug: # kpts in the curr BB: "<<boundingBox.keypoints.size()<<endl;
    //cout<<"debug: average kpts distance: "<<averageDistance<<endl;

    for (int i = to_delete.size()-1; i> 0; --i)
    {
        boundingBox.keypoints.erase(boundingBox.keypoints.begin()+to_delete[i]);
        boundingBox.kptMatches.erase(boundingBox.kptMatches.begin()+to_delete[i]);
    }
    //cout<<"debug: # kpts in the curr BB after filtering: "<<boundingBox.keypoints.size()<<endl;

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
    double dT = 1.0/frameRate; 
    std::vector<double> distRatios;

    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); ++it1)
    {
        cv::KeyPoint kpt_prev_1 = kptsPrev.at(it1->queryIdx); 
        cv::KeyPoint kpt_curr_1 = kptsCurr.at(it1->trainIdx);
        double min_dist = 100; 
        for (auto it2 =kptMatches.begin()+1; it2 != kptMatches.end(); ++it2 )
        {
            cv::KeyPoint kpt_prev_2 = kptsPrev.at(it2->queryIdx); 
            cv::KeyPoint kpt_curr_2 = kptsCurr.at(it2->trainIdx);
            double dist_prev = cv::norm(kpt_prev_1.pt - kpt_prev_2.pt);
            double dist_curr = cv::norm(kpt_curr_1.pt - kpt_curr_2.pt); 

            if (dist_prev >  std::numeric_limits<double>::epsilon() && dist_curr >= min_dist)
            {
                distRatios.push_back(dist_curr/dist_prev);
            }
        }
    }

    if ( distRatios.size() == 0)
    {
        TTC = NAN;
        return; 
    }

    std::sort(distRatios.begin(), distRatios.end()); 
    long medIdx  = std::floor(distRatios.size()/2.0); 
    double medDistRatio = distRatios.size()%2 == 0 ? (distRatios[medIdx]+distRatios[medIdx+1])/2.0 : distRatios[medIdx]; 
     
     TTC = -dT/(1-medDistRatio);






  
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxilary variables 
    double dT = 1.0/frameRate; 
    double laneWidth = 4.0; 
    double minNumLidar = 20; 
    double distDiffThresh = 0.2;

    // skip and give warinng if there a few lidarpoints dectected 
    if (lidarPointsPrev.size() < minNumLidar || lidarPointsCurr.size() < minNumLidar)
    {
        std::cout<<"Warning: Too few lidar points detected- Cannot calculate TCC"<< std::endl; 
        return;
    }

    // Find the closest lidar points from reach frame
    double minPrev = 1e8;
    double minCurr = 1e8;
    std::vector<double> lidarDistancesPrev ; 
    std::vector<double> lidarDistancesCurr ;

    for (auto it1 = lidarPointsPrev.begin(); it1 != lidarPointsPrev.end(); ++it1)
    {   
        //skip lidar pt outside of the lane
        if (abs(it1->y) > laneWidth/2)
        {
            continue;
        }
        // add lidar-x to vector calculate the average distance later 
        lidarDistancesPrev.push_back(it1->x);

        if (it1->x < minPrev)
        {
            minPrev = it1->x; 
        }
    }

    for( auto it2 = lidarPointsCurr.begin(); it2 != lidarPointsCurr.end(); ++it2)
    {
        //skip lidar pt outside of the lane
        if (abs(it2->y) > laneWidth/2)
        {
            continue;
        }
        // add lidar-x to vector calculate the average distance later 
        lidarDistancesCurr.push_back(it2->x);

        if (it2->x < minPrev)
        {
            minCurr = it2->x; 
        }

    }

    double averageDistPrev = accumulate(lidarDistancesPrev.begin(), lidarDistancesPrev.end(), 0.0)/lidarDistancesPrev.size();
    double averageDistCurr = accumulate(lidarDistancesCurr.begin(), lidarDistancesCurr.end(), 0.0)/lidarDistancesCurr.size();

    //cout<<accumulate(lidarDistancesPrev.begin(), lidarDistancesPrev.end(), 0)<<endl;
    //cout<<lidarDistancesPrev.size()<<endl; 

    //std::cout<<"Debug TCC: minPrev, avePrev:  "<<minPrev<<", "<<averageDistPrev<<std::endl; 
    //std::cout<<"Debug TCC: minCurr, aveCurr:  "<<minCurr<<", "<<averageDistCurr<<std::endl;

    // If the average distance is too greater than the minimum distance (above a threshold) then use the average distance
    if ((averageDistPrev - minPrev)> distDiffThresh)
    {
        minPrev = averageDistPrev;
    }

     if ((averageDistCurr - minPrev)> distDiffThresh)
    {
        minCurr = averageDistCurr;
    }

    // median lidar pts 
    std::sort(lidarDistancesPrev.begin(), lidarDistancesPrev.end()); 
    std::sort(lidarDistancesCurr.begin(), lidarDistancesCurr.end()); 

    double median_dist_prev  = lidarDistancesPrev[lidarDistancesPrev.size()/2]; 
    double median_dist_curr =lidarDistancesCurr[lidarDistancesCurr.size()/2]; 
  

    // Apply TCC formula:
    // TTC = minCurr * dT/(minPrev -minCurr); 
    TTC = median_dist_curr * dT/(median_dist_prev-median_dist_curr);

}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // create a 2d matrix and treat it as a table to keep track the pari of bounding box share same matched kpt

    int matchedTable [prevFrame.boundingBoxes.size()][currFrame.boundingBoxes.size()]; 
    memset(matchedTable, 0, sizeof(matchedTable)); 
    

    // check boxID of each frame 
    

    for (auto match :matches)
    {
        cv::KeyPoint kpt_prev = prevFrame.keypoints[match.queryIdx]; 
        cv::KeyPoint kpt_curr = currFrame.keypoints[match.trainIdx];

        std::vector<int> prevBoxIdx, currBoxIdx; 

        for (auto it1 = prevFrame.boundingBoxes.begin(); it1 != prevFrame.boundingBoxes.end(); ++it1)
            {
                if ( it1->roi.contains(kpt_prev.pt))
                {
        
                    prevBoxIdx.push_back(it1->boxID);
                }
            }

        for (auto it2 = currFrame.boundingBoxes.begin(); it2 != currFrame.boundingBoxes.end(); ++it2)
            {
                if ( it2->roi.contains(kpt_curr.pt))
                {
                
                    currBoxIdx.push_back(it2->boxID);
                }
            }
        
        // update matched table
        for( auto id1 : prevBoxIdx)
        {
            for (auto id2 : currBoxIdx)
            {
                matchedTable[id1][id2]++; 
            }
        }

    }

    // find the best matches BB
     
    int bb_id_prev, bb_id_curr; 
  
    for(int i = 0; i < prevFrame.boundingBoxes.size(); i++)
    {
        int maxCnt  = 0;
        for (int j = 0; j < currFrame.boundingBoxes.size(); j++)
        {
            if (matchedTable[i][j] > maxCnt)
            {
                bb_id_prev = i;
                bb_id_curr = j;
                maxCnt = matchedTable[i][j];
            }
        }
        bbBestMatches.insert({bb_id_prev, bb_id_curr});

    }

    //debug
    /*
    cout<<"debug: updated matched table:"<<endl;
     for(int i = 0; i < prevFrame.boundingBoxes.size(); i++)
    {
        cout<<"[ "; 
        for (int j = 0; j < currFrame.boundingBoxes.size(); j++)
        {
            cout<< matchedTable[i][j]<< " ";
   
        }
        cout<<" ]"<<endl;

    }
    

    cout<<"debug: number bbestMaches:  "<< bbBestMatches.size()<<endl;  
    for(auto it = bbBestMatches.begin() ; it != bbBestMatches.end(); it++)
    {
        cout<< "debug: show mathBB: first: "<< it->first<< ", second: "<< it->second<< endl;

    }
    */

}
