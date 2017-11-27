#pragma once
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

struct ignore
{
	Point2d prev;
	Point2d cur;
	bool flag;
	ignore(Point2d pe, Point2d ce, bool fl)
	{
		cur = ce;
		prev = pe;
		flag = fl;
	}
};
enum Detection_algo { BACKGROUND_SUB, IMPROVED_BG, NEW_MODEL };
class ActivityDetection
{
	Detection_algo use;
	double area_thresh;
	double proximity;
	int prev_upd_rate;
	string videoFilename; //full path

public:
	ActivityDetection(string video , Detection_algo d , double area = 250, double prox = 30, int upd_rate = 10);
	void init(); //ROI initialisation
	void Run();
	void Naive_BGSubtraction();
	void Improved_BGSubtraction();
	void NewModel();
	bool refineSegments(Mat, Mat& img);
	bool exist(vector< vector<Point> > , Point2f );
	bool nearby(Point2d fin, Point2d init, double e);
	~ActivityDetection();
};

