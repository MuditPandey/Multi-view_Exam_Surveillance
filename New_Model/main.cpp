#include <iostream>
#include "ActivityDetection.h"
//#include <dirent.h>
//#include <unistd.h>
#include <iostream>
#include <vector>
#include <stdbool.h>
#include <sstream>
#include <fstream>
#include <math.h>
using namespace std;

int main()
{
	string s;
	string path("./Videos/");
	//Read video files from folder
	string file=path+"G106_c2_appended2min.mp4";
	cout<<"Running code on video file: "<<file<<endl;
	ActivityDetection *detect = new ActivityDetection(file,IMPROVED_BG); //IMPROVED_BG BACKGROUND_SUB
	//detect->Run();
	detect->TestCameraConfig(SIFT_MOD);
	return 0;
}
