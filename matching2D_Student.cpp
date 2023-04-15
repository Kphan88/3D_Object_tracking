#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                    std::vector<cv::KeyPoint> &kPtsRef,
                    cv::Mat &descSource, 
                    cv::Mat &descRef,
                    std::vector<cv::DMatch> &matches, 
                    std::string descriptorType, 
                    std::string matcherType, 
                    std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_L2;//cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // Fix bug
        if (descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);         
        }

        // create matcher 
        cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
          
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knnMatches; 

        double t = (double)cv::getTickCount(); 

        matcher->knnMatch(descSource, descRef, knnMatches, 2);
        double endTime = (double)cv::getTickCount();
        
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout <<"KNN with N =2 matches in " << 1000 * t / 1.0 << " ms" << std::endl;

        // Filter matches uing distance ration between the best and second matches

        double minDistRatio = 0.8; 
        for (auto it = knnMatches.begin(); it != knnMatches.end(); it++)
        {
            if ((*it)[0].distance < minDistRatio*(*it)[1].distance)
                matches.push_back((*it)[0]);
        } 

        std::cout<<"Distance Ratio removed: "<<knnMatches.size() - matches.size()<<" knn matches"<< std::endl; 
        std::cout<<"Number of matches after KNN:                            "<< matches.size()<< std::endl; 


    }

}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF")==0)
    {
        // Initilaize BRIEF feature dectector
        int bytes = 32; 
        bool bOrientation = false; 

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, bOrientation);

    }
    else if (descriptorType.compare("ORB")==0)
    {
        int nfeatures = 500; 
        float scaleFactor = 1.2; 
        int nlevels = 8; 
        int edgeThreshold = 31; 
        int firstLevel = 0; 
        int WTA_K = 2; 
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE; 
        int patchSize = 31; 
        int fastThreshold = 20; 

        extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    }
    else if (descriptorType.compare("FREAK")==0)
    {
        extractor = cv::xfeatures2d::FREAK::create(); 
    }
    else if (descriptorType.compare("AKAZE")==0)
    {
        extractor = cv::AKAZE::create(); 
    }
    else if (descriptorType.compare("SIFT")==0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else 
    {
    std::cerr<<"Invadlid descriptor"<< std::endl; 
    exit(-1);
    }
   


    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //std::cout << descriptorType << " descriptor extraction in       " << 1000 * t / 1.0 << " ms" << std::endl;
    //std::cout<<"extract "<< descriptors.size()<< std::endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


// Detect keypoints in image using the traditional HARRIS detector

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int blockSize =2; 
    int apertureSize = 3; 
    double maxOverlap =0;
    int minResponse = 100; 
    double k = 0.04; 

    // start time
    double t = (double)cv::getTickCount(); 

    cv::Mat dst, dst_norm, dst_norm_scaled; 
    dst = cv::Mat::zeros(img.size(), CV_32FC1); 
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT); 
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled); 

    // Loof for prominent corner and instantiate keypoints 
    for (int i =0; i <dst_norm.rows; i++)
    {
        for (int j = 0; j < dst_norm.cols; j++)
        {
            int response = (int)dst_norm.at<float>(i,j); 
            if (response > minResponse)
            {
                cv::KeyPoint newKeyPoint; 
                newKeyPoint.pt = cv::Point2f(j, i); 
                newKeyPoint.size = 2 * apertureSize; 
                newKeyPoint.response = response;
            
                
                //Perform NMS in local neighborhood around new keypoint
                bool bOverlap = false; 
                for (auto it = keypoints.begin(); it != keypoints.end(); it++ )
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it); 
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true; 
                        if (newKeyPoint.response > (*it).response)
                        {
                            *it = newKeyPoint; 
                            break;
                        }
                    }
                }
                if (!bOverlap)
                {
                    keypoints.push_back(newKeyPoint);
                }
                
               
            }
        }
    }

    //Show runing time
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "HARRIS detection with n=" << keypoints.size() << " keypoints in        " << 1000 * t / 1.0 << " ms" << endl;

    // Visualize keypoints 
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}


// Detect keypoints in image using the modern detector

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    // start time
    double t = (double)cv::getTickCount();

    cv::Ptr<cv::FeatureDetector> detector;

    if (detectorType.compare("FAST")==0)
    {
        int threshold = 30; 
        bool bNMS = true; 
        cv::FastFeatureDetector::DetectorType detType =  cv::FastFeatureDetector::TYPE_9_16; 

        detector = cv::FastFeatureDetector::create(threshold, bNMS, detType); 
    } 
    else if ((detectorType.compare("BRISK")==0))
    {
        int threshold = 30; 
        int octave = 3; 
        float patterScale = 1.0f; 

        detector = cv::BRISK::create(threshold, octave, patterScale);
    }
    else if ((detectorType.compare("ORB")==0))
    {
        int nfeatures = 500; 
        float scaleFactor = 1.2; 
        int nlevels = 8; 
        int edgeThreshold = 31; 
        int firstLevel = 0;
        int WTA_k = 2; 
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
        int patchSize = 31; 
        int fastThreshold = 20;

        detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel , WTA_k, scoreType, patchSize, fastThreshold);
    }
    else if((detectorType.compare("AKAZE")==0))
    {
        detector = cv::AKAZE::create(); 
    }
    else if ((detectorType.compare("SIFT")==0))
    {
        detector = cv::xfeatures2d::SIFT::create(); 
    }

    // Detect keypoints N
    detector->detect(img, keypoints);

    //Show runing time
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in      " << 1000 * t / 1.0 << " ms" << endl;

      // Visualize keypoints 
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}   



