#ifndef ONED_RANGE_H
#define ONED_RANGE_H
#include "node.h"

class oned_range
{
	public:
	static node* splitnd(node *,double,double);
    static void rangequery(node *,double,double);
    static void mkrangetree(node *);  
};

#endif

