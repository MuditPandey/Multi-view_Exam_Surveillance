#include "ActivityDetection.h"
//opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//C++
#include <iostream>
#include <vector>
#include <stdbool.h>
#include <string>
#include <sstream>
#include <fstream>
#include <math.h>

using namespace cv;
using namespace std;

vector<int> frame_numbers;
vector<double> frame_time;
vector<ignore> ignore_set;
Mat frame;

ActivityDetection::ActivityDetection(string video, Detection_algo d, double area = 250, double prox = 30, int upd_rate = 10)
{
	videoFilename=video;
	use = d;
	area_thresh = area;
	proximity = prox;
	prev_upd_rate = upd_rate;
}

void ActivityDetection::Run()
{
	switch (use) {
	case BACKGROUND_SUB: Naive_BGSubtraction();
		break;
	case IMPROVED_BG: Improved_BGSubtraction();
		break;
	case NEW_MODEL: NewModel();
		break;
	default:
		std::cout << "Incorrect parameter for activity detection" << endl;
	}
}

void ActivityDetection::Naive_BGSubtraction()
{
}

void ActivityDetection::Improved_BGSubtraction()
{
	
	Mat fgMaskMOG2;
	Ptr<BackgroundSubtractorMOG2> b;
	
	int pre = 1-prev_upd_rate;
	int keyboard;
	Mat prev_frame;
	VideoCapture capture(videoFilename);
	int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));
	if (!capture.isOpened()) {
		//error in opening the video input
		cerr << "Unable to open video file: " << videoFilename << endl;
		return;
		//exit(EXIT_FAILURE);
	}
	//char EXT[] = { (char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0 };
	//cout << EXT;
	Size S = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH), (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	VideoWriter outputVideo;
	outputVideo.open("Entire_model2.avi", CV_FOURCC('X', 'V', 'I', 'D'), capture.get(CV_CAP_PROP_FPS), S, true);
	if (!outputVideo.isOpened())
	{
		cout << "Could not open the output video for write: " << endl;
		//return;
	}
	//read input data. ESC or 'q' for quitting
	while ((char)keyboard != 'q' && (char)keyboard != 27) {
		//read the current frame
		if (!capture.read(frame)) {
			cerr << "Unable to read next frame.2" << endl;
			cerr << "Exiting..." << endl;
			//exit(EXIT_FAILURE);
			break;
		}
		//if (capture.get(CAP_PROP_POS_FRAMES) > 200)
		//break;
		//*****
		//Mat mogframe(frame);
		Mat cur_frame = frame.clone();
		Mat cur_frame_gs, prev_frame_gs, diff_frame;
		if (capture.get(CAP_PROP_POS_FRAMES) > 1)
		{
			cvtColor(cur_frame, cur_frame_gs, COLOR_BGR2GRAY);
			//imshow("cur frame",cur_frame_gs);
			cvtColor(prev_frame, prev_frame_gs, COLOR_BGR2GRAY);
			//imshow("prev frame", prev_frame_gs);
			subtract(prev_frame_gs, cur_frame_gs, diff_frame);
			//imshow("diff", diff_frame);
			Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
			morphologyEx(diff_frame, diff_frame, MORPH_ERODE, element2, Point(-1, -1), 3);
			threshold(diff_frame, diff_frame, 15, 255, THRESH_BINARY);
			//imshow("diff", diff_frame);
		}
		b->apply(frame, fgMaskMOG2, 0.008);
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_DILATE, element);
		morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_ERODE, element, Point(-1, -1), 2);
		//element = getStructuringElement(MORPH_RECT, Size(11, 11));
		//morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_CLOSE, element, Point(-1, -1), 1);
		if (pre + prev_upd_rate == capture.get(CAP_PROP_POS_FRAMES))
		{
			prev_frame = frame.clone();
			pre += prev_upd_rate;
		}
		if (refineSegments(diff_frame, fgMaskMOG2) == true)
		{
			frame_numbers.push_back((int)capture.get(CAP_PROP_POS_FRAMES));
			frame_time.push_back(capture.get(CAP_PROP_POS_MSEC));
		}
		//get the frame number and write it on the current frame*/

		stringstream ss;
		ss << capture.get(CAP_PROP_POS_FRAMES);
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		string frameNumberString = ss.str() + " MOG2";
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
		//show the current frame and the fg masks
		outputVideo << frame;
		imshow("FG Mask MOG 2", fgMaskMOG2);
		imshow("MogFrame", frame);
		keyboard = waitKey(30);
	}
	capture.release();
}

void ActivityDetection::NewModel()
{
}


ActivityDetection::~ActivityDetection()
{
}
bool ActivityDetection::refineSegments(Mat dif, Mat &img)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours_diff;
	vector<Vec4i> hierarchy_diff;
	bool flag = false;
	Mat temp = dif.clone();
	findContours(temp, contours_diff, hierarchy_diff, RETR_LIST, CHAIN_APPROX_NONE);
	findContours(img, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);
	if (contours.size() == 0 || contours.size()>100)
	{
		cout << "empty";
		return false;
	}
	// iterate through all the top-level contours,
	vector<Rect> boundRect;
	double sum = 0;

	for(int i = 0; i < contours.size(); i++)
	{
		double area = fabs(contourArea(Mat(contours[i])));
		//	sum += area;
		Moments mu;
		mu = moments(contours[i], false);
		///  Get the mass centers:
		Point2d mc;
		mc = Point2d(mu.m10 / mu.m00, mu.m01 / mu.m00);
		if (mc.x < 50 || mc.x>550 || mc.y>400)
		{
			if (exist(contours_diff, mc) == false)
			{
				ignore nd(mc,mc, false);
				ignore_set.push_back(nd);
			}
		}
		if (exist(contours_diff, mc) == true)
			continue;
		if (area >area_thresh)
		{
			boundRect.push_back(boundingRect(Mat(contours[i])));
			//minEnclosingCircle((Mat)contours[i], center[i], radius[i]);
		}
	}

	/// Draw polygonal contour + bonding rects + circles
	if (boundRect.size() > 0)
		flag = true;
	for (int i = 0; i< boundRect.size(); i++)
	{
		//drawContours(frame, contours, i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 2, 8, 0);
		//circle(frame, center[i], (int)radius[i], Scalar(0, 0, 255), 2, 8, 0);
	}
	return flag;
	//imshow("exa", frame);
	//Scalar color(0, 0, 255);
	//drawContours(dst, contours, largestComp, color, FILLED, LINE_8, hierarchy);
}

bool ActivityDetection::exist(vector< vector<Point> > ct, Point2f ref)
{
	/*for (int i = 0; i < ct.size(); i++)
	{
	Moments mu;
	mu = moments(contours[i], false);
	///  Get the mass centers:
	Point2f mc;
	mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
	if (ref - mc <= PROXIMITY)
	{

	}
	}*/
	if (ignore_set.empty())
		return false;
	for (int i = 0; i < ignore_set.size(); i++)
	{
		if (nearby(ref, ignore_set[i].cur, proximity))
		{
			ignore_set[i].prev = ignore_set[i].cur;
			ignore_set[i].cur = ref;
			return true;
		}
	}
	return false;
}

bool ActivityDetection::nearby(Point2d fin, Point2d init, double e)
{
	if (sqrt((fin.x - init.x)*(fin.x - init.x) + (fin.y - init.y)*(fin.y - init.y)) <= e)
		return true;
	return false;
}