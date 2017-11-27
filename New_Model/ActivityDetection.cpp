#include "ActivityDetection.h"
//opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//C++
#include <iostream>
#include <vector>
#include <map>
#include <stdbool.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <sstream>
#include <fstream>
#include <math.h>
#include <float.h>

#include "Misc_utils.h"
#include "bal_bst.h"
#include "node.h"
#include "oned_range.h"

using namespace cv;
using namespace std;


int ActivityDetection::width;
int ActivityDetection::height;
vector<Point2d> ActivityDetection::ROI;
Mat ActivityDetection::ref_frame;

vector<ignore_it> ignore_set;
Mat frame;
int length_slider=0,breadth_slider=0;

ActivityDetection::ActivityDetection(string video, Detection_algo d, double area, double prox , int upd_rate)
{
	videoFilename=video;
	use_algo = d;
	area_thresh = area;
	proximity = prox;
	prev_upd_rate = upd_rate;
}
static void ROI_helper(int,void*)
{
    Mat img(ActivityDetection::get_reference_frame().clone());
    double length=((double)length_slider/100)*img.cols;
    double breadth=((double)breadth_slider/100)*img.rows;
    cout<<"length="<<length<<" breadth="<<breadth<<endl;
    Point2d center((float)img.cols/2,(float)img.rows/2);
    Point2d upper(center.x-length/2,center.y-breadth/2);
    Point2d lower(center.x+length/2,center.y+breadth/2);
    ActivityDetection::set_ROI_params(upper,lower);
    Mat show_ROI(ActivityDetection::get_reference_frame().clone());
    rectangle(show_ROI,upper,lower,Scalar(0,0,255),1);
    circle(show_ROI,center,5,Scalar(0,0,255),-1);
    circle(show_ROI,upper,5,Scalar(0,0,255),-1);
    circle(show_ROI,lower,5,Scalar(0,0,255),-1);
    imshow("ROI initialization",show_ROI);
}
Mat ActivityDetection:: get_reference_frame()
{
    return ActivityDetection::ref_frame;
}
void ActivityDetection:: set_ROI_params(Point2d up,Point2d bot)
{
    ROI.clear();
    ROI.push_back(up);
    ROI.push_back(bot);
}
Size ActivityDetection:: get_size()
{
   Size S(width,height);
   return S;
}
vector<Point2d> ActivityDetection::get_ROI()
{
    return ROI;
}
void ActivityDetection::setROI()
{
    VideoCapture cap(videoFilename);
    if(!cap.isOpened())
    {
        cout<<"Error opening video file:"<<videoFilename<<". ROI intitialization failed. Exiting.\n";
        exit(-1);
    }
    Mat frame;
    cout<<"Fetching reference frame from video file:"<<videoFilename<<endl;
    if(!cap.read(frame))
    {
        cout<<"Unable to read frame. ROI initialization failed. Exiting.";
        exit(-2);
    }
    cout<<"cols"<<frame.cols<<" rows:"<<frame.rows<<endl;
    ref_frame=frame;
    height=(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    width=(int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
    cout<<"height"<<height<<" width"<<endl;
    cap.release();
    namedWindow("ROI initialization",WINDOW_AUTOSIZE);
    createTrackbar("Length","ROI initialization",&length_slider,100,ROI_helper);
    createTrackbar("Width","ROI initialization",&breadth_slider,100,ROI_helper);
    ROI_helper(length_slider,0);
    int keyboard=waitKey();
    while((char)keyboard!=27 && (char)keyboard!='q');
    destroyAllWindows();
}
void ActivityDetection::Run()
{
	cout<<"Set region of interest(ROI)."<<endl;
	setROI();
	//while(waitKey()!=27 || (char)waitKey()!='q');
	if(ROI.size()==2)
        cout<<"Region of interest set. Rectangle with upper left corner: ("<<ROI[0].x<<","<<ROI[0].y<<") bottom right corner: ("<<ROI[1].x<<","<<ROI[1].y<<")"<<endl;
    else
        cout<<"Irregular ROI defined."<<endl;
	cout<<"Running Algorithm: "<<use_algo<<endl;
	switch (use_algo) {
	case BACKGROUND_SUB: Naive_BGSubtraction();
		break;
	case IMPROVED_BG: Improved_BGSubtraction();
		break;
	case NEW_MODEL: NewModel();
		break;
	default:
		cout <<"Incorrect parameter for activity detection" << endl;
		cout<<"Exiting.."<<endl;
	}
}

void ActivityDetection::Naive_BGSubtraction()
{
  vector<pair <double,double> > l;
  for(int i=0;i<10;i++)
  {
  	l.push_back(pair<double,double>(i,i*2+5));
  }
  graphs::plot_graph("test",l);
}

void ActivityDetection::Improved_BGSubtraction()
{
    cout<<"Running improved Background Subtraction.\n";
	map<int,vector < vector<Rect> > > draw_bounding_box;
	vector<pair<double,double> > frameScores;
	Mat fgMaskMOG2;
	Ptr<BackgroundSubtractorMOG2> bgMOG2;
	bgMOG2 = createBackgroundSubtractorMOG2(100,70.5,false);
	int pre = 1-prev_upd_rate;
	int keyboard;
	Mat prev_frame;
	cout<<"Opening Video file:"<<videoFilename<<endl;
	VideoCapture capture(videoFilename);
	//int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));
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
		cerr << "Could not open the output video for write: " << endl;
		return;
		//return;
	}
	//read input data. ESC or 'q' for quitting
	while ((char)keyboard != 'q' && (char)keyboard != 27) 
	{
		//read the current frame
		if (!capture.read(frame)) {
			cerr << "Unable to read next frame." << endl;
			cerr << "Exiting..." << endl;
			//exit(EXIT_FAILURE);
			break;
		}
		//if (capture.get(CAP_PROP_POS_FRAMES) > 200)
		//break;
		//*****
		//Mat mogframe(frame);
		Mat cur_frame = frame.clone();
		//fastNlMeansDenoisingColored(cur_frame,cur_frame);
		Mat cur_frame_gs, prev_frame_gs, diff_frame;
		if (capture.get(CAP_PROP_POS_FRAMES) > 1)
		{
			cvtColor(cur_frame, cur_frame_gs, COLOR_BGR2GRAY);
			imshow("cur frame",cur_frame_gs);
			cvtColor(prev_frame, prev_frame_gs, COLOR_BGR2GRAY);
			imshow("prev frame", prev_frame_gs);
			subtract(prev_frame_gs,cur_frame_gs,diff_frame);
			imshow("diff", diff_frame);
			Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
			morphologyEx(diff_frame, diff_frame, MORPH_ERODE, element2, Point(-1, -1), 3);
			threshold(diff_frame, diff_frame, 15, 255, THRESH_BINARY);
			imshow("diff", diff_frame);
		}
		bgMOG2->apply(frame, fgMaskMOG2, 0.008);
		Mat e_element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
		morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_ERODE, e_element, Point(-1, -1), 1);
		Mat d_element = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
		morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_DILATE, d_element,Point(-1, -1), 2);

		//element = getStructuringElement(MORPH_RECT, Size(11, 11));
		//morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_CLOSE, element, Point(-1, -1), 1);
		if (pre + prev_upd_rate == capture.get(CAP_PROP_POS_FRAMES))
		{
			prev_frame = frame.clone();
			pre += prev_upd_rate;
		}

		/*if (refineSegments(diff_frame, fgMaskMOG2) == true)
		{
			frame_numbers.push_back((double)capture.get(CAP_PROP_POS_FRAMES));
			frame_time.push_back(capture.get(CAP_PROP_POS_MSEC));
		}*/

		//get the frame number and write it on the current frame*/
		vector<Rect> boundRect,ignored,too_big;
        double score=refineSegments(diff_frame, fgMaskMOG2,boundRect,ignored,too_big);
        int frame_no=capture.get(CAP_PROP_POS_FRAMES);
        vector< vector<Rect> > add;
        add.push_back(boundRect);
        add.push_back(ignored);
        add.push_back(too_big);
        draw_bounding_box.insert(pair<int,vector< vector<Rect> > > (frame_no,add));
        frameScores.push_back(pair<int,double>(frame_no,score));
        rectangle(frame,ROI[0], ROI[1], Scalar(0, 255, 0), 2, 8, 0);
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
	destroyAllWindows();
	//We have scores for all frames now.
	/*node* root=NULL;
	for(int i=0;i<(int)frameScores.size();i++)
	{
		root=bal_bst::insert(root,frameScores[i].second,(long int)frameScores[i].first);
	}
	cout<<"Print tree:\n";
	bal_bst::inorder(root);
	//oned_range::mkrangetree(root);
	//node::reportsubtree(root);
	cout<<"Range tree constructed with frame scores!\n";
	*/
	char ch;
	unordered_map<long int,int> out_frames;
	do
	{
		out_frames.clear();
		cout<<"Frame scores calculated! Plotting graph."<<endl;
		double score_thresh;
		graphs::plot_interactive_graph("Frame Score Plot",frameScores,score_thresh);
		cout<<"Frame score threshold set at: "<<score_thresh<<endl;
		//Fetch rangequeryired frames
		//out_frames=get_output_frames(root,(int)frameScores.size(),score_thresh);
		for(int i=0;i<(int)frameScores.size();i++)
		{
			if(frameScores[i].second > score_thresh)
			{
				if(out_frames.find(frameScores[i].first)==out_frames.end())
				{
					out_frames.insert(pair<long int,int>(frameScores[i].first,1));
				}
			}
		}
		cout<<out_frames.size()<<"/"<<frameScores.size()<<" frames are above threshold. Ratio:"<<(int)out_frames.size()/(double)frameScores.size()<<endl;
		cout<<"Change threshold?(Y/N) :";
		cin>>ch;
	}while(ch=='Y');	
	generate_output_video(out_frames,draw_bounding_box);
}

void ActivityDetection::NewModel()
{
	
}


ActivityDetection::~ActivityDetection()
{
}
double ActivityDetection::refineSegments(Mat dif, Mat img,vector<Rect> &boundRect,vector<Rect> &ignored,vector<Rect> &too_big)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours_diff;
	vector<Vec4i> hierarchy_diff;
	Mat temp = dif.clone();
	findContours(temp, contours_diff, hierarchy_diff, RETR_LIST, CHAIN_APPROX_NONE);
	findContours(img, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);
	// Ignore if no. of contours exceed threshold of 100 since no. of contours >100 most likely indicates noise
	if (contours.size() == 0 || contours.size()>100)
	{
		//cout << "-empty-";
		return false;
	}
	// iterate through all the top-level contours,
    double frameScore=0;
	for(int i = 0; i <(int)contours.size(); i++)
	{
		double area = fabs(contourArea(Mat(contours[i])));
		//	sum += area;
		if(area >10000)
		{
			too_big.push_back(boundingRect(Mat(contours[i])));
			continue;
		}
		if (area >area_thresh)
		{
			Moments mu;
			mu = moments(contours[i], false);
			//  Get the mass centers:
			Point2d mc;
			mc = Point2d(mu.m10 / mu.m00, mu.m01 / mu.m00);
			//Replace with ROI
			ignore_it* exist=in_ignore_set(mc,area);
			if (!ROI_intersect(mc) && exist==NULL)
			{
				cout<<"Adding element in ignore set found at ("<<mc.x<<","<<mc.y<<").\n";
				ignore_it new_contour(mc,mc,area,true);
				ignore_set.push_back(new_contour);
				ignored.push_back(boundingRect(Mat(contours[i])));
			}
			else if(ROI_intersect(mc) && exist!=NULL)
			{
				if(exist->life > 20*24)
				{
					cout<<"Element is now native to ROI\n";
					exist->flag=false;
				}
				else
				{
					cout<<"Object found in ignore which is now in ROI at ("<<mc.x<<","<<mc.y<<").Previous Location ("<<exist->prev.x<<","<<exist->prev.y<<").Increasing life.\n";
					exist->life++;
				}
				if(exist->flag==false)
				{
					boundRect.push_back(boundingRect(Mat(contours[i])));
					frameScore+=(0.7*area);
				}	

			}
			else if (!ROI_intersect(mc) && exist!=NULL)
			{
				if(exist->flag==false)
				{
					exist->flag=true;
					exist->life=0;
				}	
			}
			else if (ROI_intersect(mc) && exist==NULL)
			{
				boundRect.push_back(boundingRect(Mat(contours[i])));
				frameScore+=(0.7*area);
			}
			
			//boundRect.push_back(boundingRect(Mat(contours[i])));
			//minEnclosingCircle((Mat)contours[i], center[i], radius[i]);
		}
	}	

	for(int i = 0; i<(int)boundRect.size(); i++)
	{
		//drawContours(frame, contours, i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 2, 8, 0);
		//circle(frame, center[i], (int)radius[i], Scalar(0, 0, 255), 2, 8, 0);
	}
	for(int i = 0; i<(int)ignored.size(); i++)
	{
		//drawContours(frame, contours, i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(frame, ignored[i].tl(), ignored[i].br(), Scalar(255, 0, 0), 2, 8, 0);
		//circle(frame, center[i], (int)radius[i], Scalar(0, 0, 255), 2, 8, 0);
	}
	for(int i = 0; i<(int)too_big.size(); i++)
	{
		//drawContours(frame, contours, i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(frame, too_big[i].tl(), too_big[i].br(), Scalar(0, 255, 255), 2, 8, 0);
		//circle(frame, center[i], (int)radius[i], Scalar(0, 0, 255), 2, 8, 0);
	}
	return frameScore;
	//imshow("exa", frame);
	//Scalar color(0, 0, 255);
	//drawContours(dst, contours, largestComp, color, FILLED, LINE_8, hierarchy);
}

ignore_it* ActivityDetection::in_ignore_set(Point2d refer,double area)
{

	if (ignore_set.empty())
		return NULL;
	for (int i = 0; i <(int)ignore_set.size(); i++)
	{
		if (nearby(refer, ignore_set[i].cur,ignore_set[i].approx_area,area,proximity,400))
		{
			ignore_set[i].prev = ignore_set[i].cur;
			ignore_set[i].cur = refer;
			ignore_set[i].approx_area=area;
			return &ignore_set[i];		
		}
	}
	return NULL;
}

bool ActivityDetection::nearby(Point2d fin, Point2d init,double area,double new_area,double error_dist,double error_area)
{
	if (sqrt((fin.x - init.x)*(fin.x - init.x) + (fin.y - init.y)*(fin.y - init.y)) <= error_dist && abs(area-new_area)<=error_area)
		return true;
	return false;
}
bool ActivityDetection::ROI_intersect(Point2d pos)
{
	if(pos.x>ROI[0].x && pos.x<ROI[1].x && pos.y>ROI[0].y && pos.y<ROI[1].y)
		return true;
	return false;
}
vector<Point2d> ActivityDetection::find_Students()
{
	vector<Point2d> default_val;
	return default_val; 
}
void ActivityDetection::generate_output_video(unordered_map<long int,int> frames,map< int, vector< vector<Rect> > > draw_bounding_box)
{
	VideoCapture cap(videoFilename);
	VideoWriter outputVideo;
	Size S = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH), (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	string output_name=videoFilename+"_OUTPUT.avi";
	outputVideo.open(output_name, CV_FOURCC('X', 'V', 'I', 'D'), cap.get(CV_CAP_PROP_FPS), S, true);
	if (!outputVideo.isOpened())
	{
		cout << "Could not open the output video for write: " <<endl;
		exit(-1);
	}
	if(!cap.isOpened())
	{
		cout<<"Video file "<<videoFilename<<" did not open correctly. Exiting.";
		exit(-2);
	}
	Mat frame;
	int keyboard;
	while(cap.read(frame) && (char)keyboard!=27)
	{
		if(frames.find((int)cap.get(CAP_PROP_POS_FRAMES))!=frames.end())
		{
			//draw stuff on frame
			map< int ,vector< vector<Rect> > > ::iterator it=draw_bounding_box.find((int)cap.get(CAP_PROP_POS_FRAMES));
			if(it!=draw_bounding_box.end())
			{
				for(int i = 0; i<(int)it->second[0].size(); i++)
				{
					//drawContours(frame, contours, i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point());
					rectangle(frame, it->second[0][i].tl(), it->second[0][i].br(), Scalar(0, 0, 255), 2, 8, 0);
					//circle(frame, center[i], (int)radius[i], Scalar(0, 0, 255), 2, 8, 0);
				}
				for(int i = 0; i<(int)it->second[1].size(); i++)
				{
					//drawContours(frame, contours, i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point());
					rectangle(frame, it->second[1][i].tl(), it->second[1][i].br(), Scalar(255, 0, 0), 2, 8, 0);
					//circle(frame, center[i], (int)radius[i], Scalar(0, 0, 255), 2, 8, 0);
				}
				for(int i = 0; i<(int)it->second[2].size(); i++)
				{
					//drawContours(frame, contours, i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point());
					rectangle(frame, it->second[2][i].tl(), it->second[2][i].br(), Scalar(0, 255, 255), 2, 8, 0);
					//circle(frame, center[i], (int)radius[i], Scalar(0, 0, 255), 2, 8, 0);
				}
				rectangle(frame,ROI[0], ROI[1], Scalar(0, 255, 0), 2, 8, 0);
				stringstream ss;
				ss << cap.get(CAP_PROP_POS_FRAMES);
				rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
					cv::Scalar(255, 255, 255), -1);
				string frameNumberString = ss.str();
				putText(frame, frameNumberString.c_str(), cv::Point(15, 15),FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
				outputVideo<<frame;	
			}
			else
			{
				cout<<"Frame data not found!\n";
			}
			
		}
	 	keyboard=waitKey(30);
	}
	cap.release();
}

/*unordered_map<long int,int> ActivityDetection::get_output_frames(node* root,int total,double score_thresh)
{
	cout<<"Getting output frames!\n";
	unordered_map<long int,int> ret_frames;
	oned_range::rangequery(root,score_thresh,DBL_MAX,ret_frames);
	cout<<ret_frames.size()<<"/"<<total<<" frames are above threshold. Ratio:"<<(int)ret_frames.size()/(double)total<<endl;
	return ret_frames;
}*/
struct mouse_struct
{
	vector<Point2d> pts;
	Mat fr;
	mouse_struct(Mat b)
	{
		fr=b;
	}
};
Mat og;
static void Mouse_action(int event,int x,int y,int flag,void *data)
{
	//cout<<"Mouse at: ("<<x<<","<<y<<")"<<endl;
	mouse_struct *addo=(mouse_struct *)data;
	if(event==EVENT_LBUTTONDOWN)
	{
		Point2d p(x,y);
		cout<<"Adding Point "<<p<<endl;
		int flag=0;
		for(int i=0;i<(int)addo->pts.size();i++)
		{	if(addo->pts.at(i)==p)
			{
				flag=1;
				break;
			}
		}
		if(!flag)
			addo->pts.push_back(p);
		circle(addo->fr,p,3,Scalar(0,0,255),-1);
		Point2d text_pos;
		if(p.x-5>0)
			text_pos.x=p.x-5;
		else
			text_pos.x=p.x+5;
		if(p.y-5>0)
			text_pos.y=p.y-5;
		else
			text_pos.x=p.y+5;
		stringstream s;
		s<<(int)addo->pts.size();
		putText(addo->fr, s.str().c_str(),text_pos,FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
	}
	else if(event==EVENT_RBUTTONDOWN)
	{
		for(int i=0;i<(int)addo->pts.size();i++)
		{	if((addo->pts.at(i).x-x)*(addo->pts.at(i).x-x)+(addo->pts.at(i).y-y)*(addo->pts.at(i).y-y)<=150)
			{
				cout<<"Deleting point at "<<addo->pts.at(i)<<endl;
				addo->pts.erase(addo->pts.begin()+i);
				addo->fr=og.clone();
				for(int i=0;i<(int)addo->pts.size();i++)
				{	
		 			circle(addo->fr,addo->pts.at(i),3,Scalar(0,0,255),-1);
		 			Point2d text_pos;
					if(addo->pts.at(i).x-50>0)
						text_pos.x=addo->pts.at(i).x-5;
					else
						text_pos.x=addo->pts.at(i).x+5;
					if(addo->pts.at(i).y-50>0)
						text_pos.y=addo->pts.at(i).y-5;
					else
						text_pos.x=addo->pts.at(i).y+5;
					stringstream s;
					s<<i+1;
					putText(addo->fr, s.str().c_str(),text_pos,FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
				}
				break;
			}
		}
	}
	imshow("Add Points",addo->fr);
} 
void ActivityDetection::drawEpilines(Mat img1,Mat img2,vector<Point3d> line,vector<Point2d> pts1,vector<Point2d> pts2)
{
	Size s=img1.size();
	RNG rng(0);
	for(int i=0;i<(int)line.size();i++)
	{
		Point2d x0(0,-line[i].z/line[i].y);
		Point2d x1(s.width,-(line[i].x*s.width+line[i].z)/line[i].y);
		//cout<<x0<<" "<<x1<<endl;
		Point2d text_pos;
		Scalar clr(rng(256),rng(256),rng(256));
		//Mark point in image 1
		circle(img1,pts1[i],3,clr,-1);
		if(pts1[i].x-50>0)
			text_pos.x=pts1[i].x-5;
		else
			text_pos.x=pts1[i].x+5;
		if(pts1[i].y-50>0)
		    text_pos.y=pts1[i].y-5;
	    else
			text_pos.x=pts1[i].y+5;
		stringstream s1;
		s1<<i+1;
		putText(img1, s1.str().c_str(),text_pos,FONT_HERSHEY_SIMPLEX, 0.5, clr);
		//Mark corresponding in image 2 and draw epilines
		circle(img2,pts2[i],3,clr,-1);
		if(pts2[i].x-50>0)
			text_pos.x=pts2[i].x-5;
		else
			text_pos.x=pts2[i].x+5;
		if(pts2[i].y-50>0)
		    text_pos.y=pts2[i].y-5;
	    else
			text_pos.x=pts2[i].y+5;
		stringstream s2;
		s2<<i+1;
		putText(img2, s2.str().c_str(),text_pos,FONT_HERSHEY_SIMPLEX, 0.5, clr);
		cv::line(img2,x0,x1,clr);
	}
	//imshow("Frame 1",img1);
	//imshow("Frame 2",img2);
}
void ActivityDetection::TestCameraConfig(Camera_calib proc)
{

	cout<<"Lets do this!"<<endl;
	string img1="tripod_seq_01_002.jpg";
	string img2="tripod_seq_01_004.jpg";
	cout<<"Showing:"<<img1<<endl;
	Mat frame1=imread(img1);
	Mat test1=frame1.clone(),ref1=frame1.clone();
	og=frame1.clone();
	namedWindow("Add Points");
	mouse_struct m1(frame1.clone());
	setMouseCallback("Add Points",Mouse_action,(void*)&m1);
	imshow("Add Points",m1.fr);
	while(1)
	{
		//Point2d l(10,10);
		//cout<<"go";
		//imshow("Img1",frame);
		int c=waitKey(0);
		if((c&255)==27)
		{
			if(m1.pts.size()>=8)			
				break;
			else
			{
				
				cout<<"Points selected:"<<m1.pts.size()<<" Pick atleast "<<8-(int)m1.pts.size()<<" more points!"<<endl;
			}
		}
	}

	cout<<"Choose corresponding points for new view in the same order:\n";
	imshow("Image points for first image",m1.fr);
	Mat frame2=imread(img2);
	og=frame2.clone();
	Mat test2=frame2.clone(),ref2=frame2.clone();
	mouse_struct m2(frame2.clone());
	namedWindow("Add Points");
	setMouseCallback("Add Points",Mouse_action,(void*)&m2);
	imshow("Add Points",m2.fr);
	while(1)
	{
		//Point2d l(10,10);
		//cout<<"go";
		//imshow("Img1",frame);
		int c=waitKey(0);
		if((c&255)==27)
		{
			if(m2.pts.size()==m1.pts.size())			
				break;
			else if(m2.pts.size()<m1.pts.size())
			{
				
				cout<<"Points selected:"<<m2.pts.size()<<" Pick "<<(int)m1.pts.size()-(int)m2.pts.size()<<" more points!"<<endl;
			}
			else
			{
				cout<<"Extra points selected! Delete "<<(int)m2.pts.size()-(int)m1.pts.size()<<" points!"<<endl;
			}
		}
	}
	cout<<"Corresponding points found ! Calculating Fundamental matrix\n";
	Mat fundamental_mat=findFundamentalMat(m1.pts,m2.pts);
	cout<<"Fundamental matrix:\n"<<fundamental_mat<<endl;
	vector<Point3d> line2,line1;
	computeCorrespondEpilines(m2.pts,2,fundamental_mat,line1);
	computeCorrespondEpilines(m1.pts,1,fundamental_mat,line2);
	destroyAllWindows();
	Mat f1=frame1.clone(),f2=frame2.clone();
	drawEpilines(frame1,f2,line2,m1.pts,m2.pts);
	imshow("Frame 1_Reference",frame1);
	imshow("Frame 2_Epilines corresponding to Pts1",f2);
	drawEpilines(frame2,f1,line1,m2.pts,m1.pts);
	imshow("Frame 2_Reference",frame2);
	imshow("Frame 1_Epilines corresponding to Pts2",f1);
	waitKey(0);
	destroyAllWindows();
	cout<<"Select new points from first image:";
	vector<Point3d> line3;
	namedWindow("Add Points");
	mouse_struct m3(test1.clone());
	og=test1.clone();
	setMouseCallback("Add Points",Mouse_action,(void*)&m3);
	imshow("Add Points",test1);
	while(1)
	{
		//Point2d l(10,10);
		//cout<<"go";
		//imshow("Img1",frame);
		int c=waitKey(0);
		if((c&255)==27)
		{
			break;
		}
	}
	computeCorrespondEpilines(m3.pts,1,fundamental_mat,line3);
	Size s=test1.size();
	//Getting test points
	RNG rng(0);
	for(int i=0;i<(int)line3.size();i++)
	{
		Point2d x0(0,-line3[i].z/line3[i].y);
		Point2d x1(s.width,-(line3[i].x*s.width+line3[i].z)/line3[i].y);
		//cout<<x0<<" "<<x1<<endl;
		Point2d text_pos;
		Scalar clr(rng(256),rng(256),rng(256));
		//Mark point in image 1
		circle(test1,m3.pts[i],3,clr,-1);
		if(m3.pts[i].x-50>0)
			text_pos.x=m3.pts[i].x-5;
		else
			text_pos.x=m3.pts[i].x+5;
		if(m3.pts[i].y-50>0)
		    text_pos.y=m3.pts[i].y-5;
	    else
			text_pos.x=m3.pts[i].y+5;
		stringstream s1;
		s1<<i+1;
		putText(test1, s1.str().c_str(),text_pos,FONT_HERSHEY_SIMPLEX, 0.5, clr);
		cv::line(test2,x0,x1,clr);
	}
	imshow("final pts",test1);
	imshow("final epi",test2);
	waitKey(0);
	destroyAllWindows();
	if(proc==TEMPLATE)
	{
		cout<<"Patches:"<<endl;
		vector< Mat > image_patches;
		Mat ref=ref1.clone(),output_res=ref2.clone();
		cvtColor(ref,ref,COLOR_BGR2GRAY);
		Mat output_res_grey;
		cvtColor(output_res,output_res_grey,COLOR_BGR2GRAY);
		for(int i=0;i<(int)m3.pts.size();i++)
		{
			//cout<<m3.pts[i].y-20<<" "<<m3.pts[i].y+21<<" "<<m3.pts[i].x-20<<" "<<m3.pts[i].x+21<<endl;
			Mat patch(ref.clone(),Range(m3.pts[i].y-20,m3.pts[i].y+20),Range(m3.pts[i].x-20,m3.pts[i].x+20));
			Mat res;
			//imshow("patch",patch);
			matchTemplate(output_res_grey,patch,res, CV_TM_CCOEFF_NORMED);
			threshold(res, res, 0.8, 1., CV_THRESH_TOZERO);
			vector<Point2d> potential_match;
			while (true)
			{
			    double minval, maxval, threshold = 0.8;
			    Point minloc, maxloc;
			  	cv::minMaxLoc(res, &minval, &maxval, &minloc, &maxloc);

			    if (maxval >= threshold)
			    {
			       potential_match.push_back(Point2d(maxloc.x+patch.cols/2,maxloc.y+patch.rows/2));
			        /*cv::rectangle(
			            ref,
			            maxloc,
			            cv::Point(maxloc.x + patch.cols, maxloc.y + patch.rows),
			            CV_RGB(0,255,0), 2
			        );*/
			        cv::floodFill(res, maxloc, cv::Scalar(0), 0, cv::Scalar(.1), cv::Scalar(1.));
			    }
			    else
			        break;
			}
			if(potential_match.size()==0)
				cout<<"No match found for point :"<<m3.pts[i]<<endl;
			Point2d match_pt;
			double min_dist=10000000;
			double DIST_THRESH=50;
			for(int j=0;j<(int)potential_match.size();j++)
			{
				double dist=abs(line3[i].x*potential_match[j].x+line3[i].y*potential_match[j].y+line3[i].z)/sqrt(line3[i].x*line3[i].x+line3[i].y*line3[i].y);
				if(dist<=DIST_THRESH && dist<min_dist)
				{
					min_dist=dist;
					match_pt=potential_match[j];
				}
			}
			if(min_dist==10000000)
			{
				cout<<"No match found for the point: "<<m3.pts[i]<<endl;
				continue;
			}
			circle(test2,match_pt,3,Scalar(0,0,255),-1);
			Point2d text_pos;
			if(match_pt.x-50>0)
				text_pos.x=match_pt.x-5;
			else
				text_pos.x=match_pt.x+5;
			if(match_pt.y-50>0)
			    text_pos.y=match_pt.y-5;
		    else
				text_pos.x=match_pt.y+5;
			stringstream s1;
			s1<<i+1;
			putText(test2, s1.str().c_str(),text_pos,FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255));

		}
		imshow("POINTS",m3.fr);
		imshow("MATCH POINTS",test2);
		imshow("EPI_REF",test1);
		waitKey(0);
	}
	else if(proc==COL_HOG)
	{
		
		Mat first=ref1.clone(),second=ref2.clone();
		Mat padded_second;
		cout<<"Padding image..\n";
		int padding =50;
		int patch_size=20;
		copyMakeBorder(second,padded_second,padding, padding, padding,padding,BORDER_CONSTANT, Scalar(0));
		cout<<"Padded!\n";
		float rrange[]={0,256};
		float grange[]={0,256};
		float brange[]={0,256};
		const float* histRange[]={rrange,grange,brange};
		const int sizes[]={256,256,256};
		const int channels[]={0,1,2};
		bool uniform=true,accumulate=false;
		//vector<Point2d> potential_match;
		for(int i=0;i<(int)m3.pts.size();i++)
		{
			cout<<"Acquiring reference patch!\n";
			Mat patch(first.clone(),Range(m3.pts[i].y-patch_size/2,m3.pts[i].y+patch_size/2),Range(m3.pts[i].x-patch_size/2,m3.pts[i].x+patch_size/2));
			cout<<"Refernce patch size: "<<patch.size()<<endl;
			
			//Color Histogram
			Mat hist_1;
			calcHist(&patch, 1, channels, Mat(), hist_1, 3, sizes, histRange, uniform, accumulate );
			normalize(hist_1, hist_1, 0, 1, NORM_MINMAX, -1, Mat() );
			
			//HOG descriptor
			vector<float> features1;
   			vector<Point> locations1;
			vector<Point2d> end_point;
			HOGDescriptor *hog1 = new HOGDescriptor();
			hog1->compute(patch,features1,Size(32,32), Size(0,0),locations1);

			Size s=first.size();
			double y_x0=-line3[i].z/line3[i].y,x_y0=-line3[i].z/line3[i].x,x_ymax= -(line3[i].z+line3[i].y*s.height)/line3[i].x,y_xmax=-(line3[i].z+line3[i].x*s.width)/line3[i].y;
			if(y_x0 >=0 && y_x0<=s.height)
				end_point.push_back(Point2d(0,y_x0));
			if(x_y0 >=0 && x_y0<=s.width)
				end_point.push_back(Point2d(x_y0,0));
			if(y_xmax >=0 && y_xmax<=s.height)
				end_point.push_back(Point2d(s.width,y_xmax));
			if(x_ymax >=0 && x_ymax<=s.width)
				end_point.push_back(Point2d(x_ymax,s.height));
			cout<<"Endpoints gathered! Size= "<<end_point.size()<<" ";
			sort (end_point.begin(), end_point.end(), [](Point2d const &a, Point2d const &b) { return a.x< b.x; });
			for(int i=0;i<(int)end_point.size();i++)
			{
					cout<<end_point[i]<<" ";
			}
			Point2d start(end_point[0]);
			double min_diff=-10000;
			double SSE_sum=100000;
			Point2d potential_match,potential_match2(0,0);
			while(start.x<=end_point[1].x)
			{
				//cout<<"Checking Epiline!\n";
				//cout<<padded_second.size();
				//cout<<start.y-20<<" "<<start.y+20<<" "<<start.x-20<<" "<<start.x+20<<endl;
				Mat fpatch(padded_second,Range(start.y-patch_size/2+padding,start.y+patch_size/2+padding),Range(start.x-patch_size/2+padding,start.x+patch_size/2+padding));
				//cout<<"Got patch"<<endl;
				Mat hist_2;
				//For 2nd view
				calcHist(&fpatch, 1, channels, Mat(), hist_2, 3, sizes, histRange, uniform, accumulate );
				normalize(hist_2, hist_2, 0, 1, NORM_MINMAX, -1, Mat() );
				for( int i = 0; i < 4; i++ )
			   	{ 
				     double diff = compareHist( hist_1, hist_2, i);
				     //cout<<"Method ["<<i<<"]. Difference= "<<diff<<endl;
				     if(i==0 && min_diff<diff)
				     {
				     	min_diff=diff;
				     	potential_match=start;
				     }
			   	}
			   	
			   	//HOG--Shift HOG from edge(TO DO)
   				vector<float> features2;
   				vector<Point> locations2;
   				HOGDescriptor *hog2 = new HOGDescriptor();
   				hog2->compute(fpatch,features2,Size(32,32), Size(0,0),locations2);
				double sum=0;
				if(features1.size()!=features2.size())
					cout<<"Bad HOG.\n";
				else
				{	
					for(int i=0;i<(int)features1.size();i++)
					{
						float v=features1[i]-features2[i];
						sum+=(v*v);
					}
					sum=sqrt(sum);
				}
				if(sum<SSE_sum)
				{
					SSE_sum=sum;
					potential_match2=start;
				}
				start.x=start.x+1;
				start.y=-(line3[i].z+line3[i].x*start.x)/line3[i].y;	
			}
			circle(test2,potential_match,3,Scalar(0,0,255),-1);
			circle(test2,potential_match2,3,Scalar(0,0,255),-1);
			Point2d text_pos;
			if(potential_match.x-50>0)
				text_pos.x=potential_match.x-5;
			else
				text_pos.x=potential_match.x+5;
			if(potential_match.y-50>0)
			    text_pos.y=potential_match.y-5;
		    else
				text_pos.x=potential_match.y+5;
			stringstream s1;
			s1<<i+1;
			putText(test2, (s1.str()+"c").c_str(),text_pos,FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255));
			if(potential_match2.x-50>0)
				text_pos.x=potential_match2.x-5;
			else
				text_pos.x=potential_match2.x+5;
			if(potential_match2.y-50>0)
			    text_pos.y=potential_match2.y-5;
		    else
				text_pos.x=potential_match2.y+5;
			putText(test2, s1.str().c_str(),text_pos,FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255));
			cout<<endl;		
		}   
		imshow("POINTS",m3.fr);
		imshow("MATCH POINTS",test2);
		imshow("EPI_REF",test1);
		waitKey(0);
		cout<<"End!\n";
	}
	else if(proc==SIFT_MOD)
	{
		Mat first=ref1.clone(),second=ref2.clone();
		Mat padded_second;
		//cout<<"Padding image..\n";
		//int padding =65;
		int patch_size=60;
		//copyMakeBorder(second,padded_second,padding, padding, padding,padding,BORDER_CONSTANT, Scalar(0));
		//cout<<"Padded!\n";
		//vector<Point2d> potential_match;
		for(int i=0;i<(int)m3.pts.size();i++)
		{
			cout<<"Acquiring reference patch!\n";
			Mat patch(first.clone(),Range(m3.pts[i].y-patch_size/2,m3.pts[i].y+patch_size/2),Range(m3.pts[i].x-patch_size/2,m3.pts[i].x+patch_size/2));
			cvtColor(patch, patch, CV_BGR2GRAY);
			cout<<"Refernce patch size: "<<patch.size()<<endl;
			Ptr<Feature2D> f_ref = xfeatures2d::SIFT::create();
			vector<KeyPoint> key_points_ref;
			Mat des_ref;
			f_ref->detectAndCompute(patch,Mat(),key_points_ref,des_ref);
			if(des_ref.empty())
			{
				cout<<"SIFT descriptor empty for reference patch "<<i+1<<" .No features found. Skipping point..\n";
				continue;
			}
			vector<Point2d> end_point;
			Size s=first.size();
			double y_x0=-line3[i].z/line3[i].y,x_y0=-line3[i].z/line3[i].x,x_ymax= -(line3[i].z+line3[i].y*s.height)/line3[i].x,y_xmax=-(line3[i].z+line3[i].x*s.width)/line3[i].y;
			if(y_x0 >=0 && y_x0<=s.height)
				end_point.push_back(Point2d(0,y_x0));
			if(x_y0 >=0 && x_y0<=s.width)
				end_point.push_back(Point2d(x_y0,0));
			if(y_xmax >=0 && y_xmax<=s.height)
				end_point.push_back(Point2d(s.width,y_xmax));
			if(x_ymax >=0 && x_ymax<=s.width)
				end_point.push_back(Point2d(x_ymax,s.height));
			cout<<"Endpoints gathered! Size= "<<end_point.size()<<" ";
			sort (end_point.begin(), end_point.end(), [](Point2d const &a, Point2d const &b) { return a.x< b.x; });
			for(int i=0;i<(int)end_point.size();i++)
			{
					cout<<end_point[i]<<" ";
			}
			
			Point2d start(end_point[0]);
			vector<DMatch> matched_des;
			vector<KeyPoint> matched_kp;
			Mat matched_patch;
			Point2d potential_match(0,0);
			//while(start.x<=end_point[1].x)
			//{
				//cout<<"Checking Epiline!\n";
				//cout<<padded_second.size();
				//cout<<start.y-20<<" "<<start.y+20<<" "<<start.x-20<<" "<<start.x+20<<endl;
				//Mat tpatch(second.clone(),Range(start.y-patch_size/2,start.y+patch_size/2),Range(start.x-patch_size/2,start.x+patch_size/2));
				//if(ip==0)
					//imshow("1st patch",tpatch);
				//cvtColor(tpatch, tpatch, CV_BGR2GRAY);
				//cout<<"Got patch"<<endl;
				Ptr<Feature2D> f_patch=xfeatures2d::SIFT::create();
				vector<KeyPoint> key_points_patch;
				Mat des_patch;
				f_patch->detectAndCompute(second.clone(),Mat(),key_points_patch,des_patch);
				FlannBasedMatcher matcher;
				vector<DMatch> matches;
				if(des_patch.empty())
				{
					cout<<"SIFT descriptor empty for test patch with x="<<start.x<<" . Features not found! Skipping point..\n";
					start.x=start.x+1;
					//Optimize this in term of slope
					start.y=-(line3[i].z+line3[i].x*start.x)/line3[i].y;	
					continue;	
				}
				matcher.match(des_ref,des_patch,matches);
				double max_dist=-1,min_dist=1000;
				for( int i = 0; i < des_ref.rows; i++ )
				{ 
					double dist = matches[i].distance;
				    if( dist < min_dist ) min_dist = dist;
				    if( dist > max_dist ) max_dist = dist;
				}
				vector<DMatch> good_matches;
				for(int i=0;i<des_ref.rows;i++)
				{
					//Used 2*min dist or some arbitary value
					if(matches[i].distance <=max(2*min_dist,0.02))
					{
						good_matches.push_back(matches[i]);
					}
				}
				
				//if(good_matches.size()>matched_des.size())
				//{
					matched_des=good_matches;
					matched_kp=key_points_patch;
					//potential_match=start;
					//matched_patch=tpatch;

				//}
				/*drawMatches( patch,key_points_ref, matched_patch, matched_kp,
              good_matches,lo, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );*/
				//start.x=start.x+20;
				//Optimize this in term of slope
				//start.y=-(line3[i].z+line3[i].x*start.x)/line3[i].y;	
			//}
			circle(test2,potential_match,3,Scalar(0,0,255),-1);
			Point2d text_pos;
			if(potential_match.x-50>0)
				text_pos.x=potential_match.x-5;
			else
				text_pos.x=potential_match.x+5;
			if(potential_match.y-50>0)
			    text_pos.y=potential_match.y-5;
		    else
				text_pos.x=potential_match.y+5;
			stringstream s1;
			s1<<i+1;
			putText(test2, (s1.str()).c_str(),text_pos,FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255));
			Mat img_match;
			drawMatches( patch,key_points_ref, second.clone(), matched_kp,
              matched_des, img_match, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
			imshow("Good matches",img_match);
			waitKey(0);
		}   
		imshow("POINTS",m3.fr);
		imshow("MATCH POINTS",test2);
		imshow("EPI_REF",test1);
		waitKey(0);
		cout<<"End!\n";
	}
}
