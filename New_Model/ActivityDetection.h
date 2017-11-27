#pragma once
//opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//C++
#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <stdbool.h>
#include <string>
#include <sstream>
#include <fstream>
#include <math.h>

#include "node.h"

using namespace cv;
using namespace std;

struct ignore_it
{
	Point2d prev;
	Point2d cur;
	double approx_area;
	int life;
	bool flag;
	ignore_it(Point2d pe, Point2d ce,double ar,bool fl,int lf=0)
	{
		prev = pe;
		cur = ce;
		approx_area=ar;
		flag = fl;
		life=lf;

	}
};
enum Detection_algo { BACKGROUND_SUB, IMPROVED_BG, NEW_MODEL };
enum Camera_calib {TEMPLATE,COL_HOG,IMPROVED_COL,VIEW_3,SIFT_MOD};
class ActivityDetection
{
	Detection_algo use_algo;
	double area_thresh;
	double proximity;
	int prev_upd_rate;
	string videoFilename; //full path

public:

    static int width;
	static int height;
	static vector<Point2d> ROI;
	static Mat ref_frame;
	static void set_ROI_params(Point2d,Point2d);
	static Size get_size();	
	static vector<Point2d> get_ROI();
	static Mat get_reference_frame();

	ActivityDetection(string video , Detection_algo d , double area = 175, double prox = 30, int upd_rate = 10);
	void setROI(); //ROI initialisation
	void Run(); //Main activity detection function
	void TestCameraConfig(Camera_calib); //Testing epipolar geometry and view integration
	void Naive_BGSubtraction();
	void Improved_BGSubtraction();
	void NewModel();
	double refineSegments(Mat,Mat,vector<Rect> &,vector<Rect> &,vector<Rect> &);
	ignore_it* in_ignore_set(Point2d,double );
	vector<Point2d> find_Students();
	bool nearby(Point2d,Point2d,double,double,double,double);
	bool ROI_intersect(Point2d);
	void generate_output_video(unordered_map<long int,int>,map< int, vector< vector<Rect> > > draw_bounding_box);
	//unordered_map<long int,int> get_output_frames(node*,int ,double );
	void drawEpilines(Mat,Mat,vector<Point3d>,vector<Point2d>,vector<Point2d>);
	~ActivityDetection();
};

