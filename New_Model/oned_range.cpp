#include "oned_range.h"
#include "node.h"
#include<iostream>
using namespace std;

/*********************************************************************************************************
 Converts the Balanced Binary Search Tree to a tree whose leaves are the values in sorted order from left
 to right
 
**********************************************************************************************************/

void oned_range::mkrangetree(node *root)
{
 if(root==NULL)
 return;
 if(root->type==0)
 return ;
 node *t2;
 node *temp;
 node *leaf=node::newnode(root->num,root->fr);
 leaf->type=0;
 if(root->left==NULL)
 {
  	root->left=leaf;
 }
 else
 {
  	temp=root->left;
	while(temp!=NULL)
 	{
 	 	t2=temp;
 	 	temp=temp->right;
 	}
  	t2->right=leaf;
 }
 mkrangetree(root->left);
 mkrangetree(root->right);
}

/*************************************************************************************************************
  The Main 1-Dimensional Range Query Function. The values in the given range are printed.

**************************************************************************************************************/

void oned_range::rangequery(node *root,double x,double y)
{
 node *vsplit=splitnd(root,x,y);
 if(node::isLeaf(vsplit))
 {
  	if(vsplit->num >=x && vsplit->num<=y)
  	cout<<vsplit->fr<<" ";
 }
 else
 {
  	node *v=vsplit->left;
  	while(!node::isLeaf(v))
  	{
   		if(x<v->num)
    	{
     		node::reportsubtree(v->right);
	 		v=v->left;
    	}
    	else
    	v=v->right;
  	 }
  if(v->num >=x && v->num<=y)
  cout<<v->fr<<" ";
  v=vsplit->right;
   while(!node::isLeaf(v))
   {
   		if(y>v->num)
    	{
     		node::reportsubtree(v->left);
		 	v=v->right;
    	}
    	else
    	v=v->left;
   }
   if(v->num >=x && v->num<=y)
   	cout<<v->fr<<" ";
  }	
}

/**************************************************************************************************************
  Function to find the split node. Returns the split node.
  
*************************************************************************************************************/

node* oned_range::splitnd(node *root,double x,double y)
{
 node *v=root;
 while(!node::isLeaf(v) &&(y<=v->num || x>v->num))
 {
  	if(y<=v->num)	
  	v=v->left;
  	else
  	v=v->right;
 }
 return v;
}

