#ifndef _Misc_utils_h
#define _Misc_utils_h_

#include <iostream>
#include <vector>

namespace graphs{
	void plot_graph(std::string title,std::vector<std::pair<double,double> > );	
	void plot_interactive_graph(std::string title,std::vector<std::pair<double,double> >,double&);
	void test();
}
namespace comp{
long int max(long int,long int);
}
#endif