#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/nonfree/nonfree.hpp"

#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <numeric>


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

    struct values
 {
  float distance;
  float slopeOfLine; 
 };

#include "distanceToBoundary.cpp"            ///////// Uncomment for calculating the distance to the boundary


values getDistanceToBoundary(Mat);



int main( int argc, char** argv )
{
  Mat frame = imread(argv[1], 1);
  values DistanceToBoundary = getDistanceToBoundary(frame);
  cout << DistanceToBoundary.distance << endl;
  cout << DistanceToBoundary.slopeOfLine << endl;
}

values getDistanceToBoundary(Mat img_scene)
{

 /////////////////////////////////////////////////////  Uncomment starting here for videocapture
/*									
    cv::VideoWriter output_cap(argv[2], 
               cap.get(CV_CAP_PROP_FOURCC),
               cap.get(CV_CAP_PROP_FPS),
               cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH),
               cap.get(CV_CAP_PROP_FRAME_HEIGHT)));
//        output_cap.open ( "outputVideo.avi");

*/
    
//        Mat frame;
//        Mat img_scene = frame;
//        cap >> frame; // get a new frame from camera
        cvtColor(img_scene, img_scene, COLOR_BGR2GRAY);
  ///////////////////////////////////////////////////// Uncomment until here for videocapture


        Mat img_object = imread("model.jpg", CV_LOAD_IMAGE_GRAYSCALE );
//        Mat img_scene = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );       ////// Comment for videocapture
        

  ///////////////////////////////////////////////////// Uncomment for erode and dilate

	// Smoothing the image

        Mat element = getStructuringElement( MORPH_RECT,Size( 2*0 + 1, 2*0+1 ),Point( 0, 0) );
        erode (img_scene,img_scene,element);
        dilate(img_scene,img_scene,element);

  //-- Step 1: Detect the keypoints using SURF Detector
  	int minHessian = 400;

  	Ptr<xfeatures2d::SURF> surf = SURF::create(minHessian);

  	vector<KeyPoint> keypoints_object, keypoints_scene;

//  SurfDescriptorExtractor extractor;

  	Mat descriptors_object, descriptors_scene;

  	equalizeHist(img_object,img_object);
  	equalizeHist(img_scene,img_scene);

  	surf->detectAndCompute(img_object, Mat(), keypoints_object, descriptors_object);
  	surf->detectAndCompute(img_scene, Mat(), keypoints_scene, descriptors_scene);

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  	FlannBasedMatcher matcher;
  	std::vector< DMatch > matches;
  	matcher.match( descriptors_object, descriptors_scene, matches );

  	double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  	for( int i = 0; i < descriptors_object.rows; i++ )
  	{ double dist = matches[i].distance;
    	if( dist < min_dist ) min_dist = dist;
    	if( dist > max_dist ) max_dist = dist;
  }

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  	std::vector< DMatch > good_matches;

 	 for( int i = 0; i < descriptors_object.rows; i++ )
  	{ if( matches[i].distance < 3*min_dist )
     		{ good_matches.push_back( matches[i]); }
  	}

      ///////////////////////////////////////////////////// Uncomment to draw matches
  	Mat img_matches;
  	drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


  //-- Localize the object
  	std::vector<Point2f> obj;
  	std::vector<Point2f> scene;

  	for( int i = 0; i < good_matches.size(); i++ )
  	{
        //-- Get the keypoints from the good matches
    	obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    	scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  	}

        ///////////////////////////////////////////////////// Filter using intensities

        float intensity_min_threshold = 0;
        float intensity_max_threshold = 20;

        for(int i = 0; i < scene.size() ; i++)
        {
 	   Scalar intensity_pixel = img_scene.at<uchar>(Point(scene[i].x, scene[i].y));
           if (intensity_pixel[0] < intensity_min_threshold || intensity_pixel[0] > intensity_max_threshold)
              {                
                scene.erase(scene.begin() + i);
                i--;
              }
        }

       
         ///////////////////////////////////////////////////// Try Catch for calculating slope and x,y intercepts to draw a line

          if(scene.size() > 100)
          { 
            //cout << "Enough points not detected" << endl;
            throw std::exception();
          }

          cv::Vec<float,4> scene_output;
          fitLine(scene, scene_output, CV_DIST_HUBER, 0, 0.01, 0.01);
        
          float slope = scene_output[1]/scene_output[0] ;
          float y_intercept = scene_output[3] - slope * scene_output[2];
          float x_intercept = -(y_intercept/slope);

          line( img_matches,Point2f(600+0,y_intercept),Point2f(600+scene_output[2],scene_output[3]), Scalar( 0, 255, 0), 2, 8, 0);
          line( img_matches,Point2f(600+x_intercept,0),Point2f(600+scene_output[2],scene_output[3]), Scalar( 0, 255, 0), 2, 8, 0);

      ///////////////////////////////////////////////////// Uncomment for getting distance to the boundary
        
        float distanceToBoundary = calculate_distance(slope, y_intercept, 1);
        std::ostringstream ss;
        ss << distanceToBoundary;
        std::string str_distance(ss.str());
        putText(img_matches, str_distance , cvPoint(100,100), FONT_HERSHEY_COMPLEX_SMALL, 5, cvScalar(0,255,0), 5, CV_AA);

      ///////////////////////////////////////////////////// Uncomment for getting distance to the boundary
               

      ///////////////////////////////////////////////////// Uncomment to draw lines connecting the pixels
       
  	for (int i =0; i<scene.size() ; i++)
   	{
    		line( img_matches,Point2f(600+scene[i].x,scene[i].y),Point2f(600+scene[i].x,scene[i].y), Scalar( 0, 255, 0), 2, 8, 0);
   	}
        
       ///////////////////////////////////////////////////// Uncomment to draw lines connecting the pixels


  	imshow( "Good Matches & Object detection", img_matches );
        waitKey(0);                                                    ////// Comment for the videocapture 'for' loop
//  	if(waitKey(10) >= 0) break;                                    ////// Uncomment for the videocapture 'for' loop

  //////////////////////////////////////////////////////////////////// Uncomment for the videocapture 'for' loop
  struct values v;
  v.distance = distanceToBoundary;
  v.slopeOfLine = slope;
  return v;
 }
