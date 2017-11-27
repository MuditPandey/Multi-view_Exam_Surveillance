#include "Misc_utils.h"
#include <fstream>
#include <vector>
#include <map>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#define GNUPLOT_ENABLE_PTY
#include "gnuplot-iostream.h"

void graphs::plot_graph(std::string title,std::vector<std::pair<double,double> > xy_pts)
{
	Gnuplot gp;
	gp<< "plot '-' with lines title '"<<title<<"'\n";
	gp.send1d(xy_pts);
}
void graphs::plot_interactive_graph(std::string title,std::vector<std::pair<double,double> > xy_pts,double &thresh)
{
	Gnuplot gp;
	double mx=0.5,my=0;
	int mb=1;
	while(mb!=3 && mb>0)
	{
		thresh=my;
		gp<<"plot '-' with lines title '"<<title<<"', [0:"<<xy_pts[(int)xy_pts.size()-1].first+1000<<"] "<<my<<" title 'threshold'\n";
		gp.send1d(xy_pts);
		gp.getMouse(mx, my, mb, "Left click to set threshold line, right click to exit.");
		//std::cout<<"You pressed mouse button "<<mb<<" at x="<<mx<<" y="<<my<<std::endl;
	}
	if(mb < 0)
		std::cout<<"Gnuplot window was closed.\n";
}
void graphs::test(){
  std::vector<std::pair<double,double>> data;
  data.emplace_back(-2,-0.8);
  data.emplace_back(-1,-0.4);
  data.emplace_back(0,-0);
  data.emplace_back(1,0.4);
  data.emplace_back(1,0.8);
  Gnuplot gp;
  gp << "plot [-5:5] sin(x) tit 'sin(x)', '-' tit 'data'\n";
  gp.send1d(data);
}  

long int comp::max(long int a, long int b)
{
    return (a > b)? a : b;	
}