#include "node.h"
#include<iostream>
#include<stdlib.h>
using namespace std;

/***************************************************************************************************************
 Function that iniatialises a new node of a binary tree
 
**************************************************************************************************************/
node* node::newnode(double x,long int f)
{
 node *temp=(node*)malloc(sizeof(node));
 temp->num=x;
 temp->fr=f;
 temp->left=NULL;
 temp->right=NULL;
 temp->parent=NULL;
 temp->type=1; 
 temp->height=1;
 return temp;
}
/**************************************************************************************************************
 Function which checks whether a node is a leaf node

**************************************************************************************************************/

int node::isLeaf(node *temp)
{
	
	if(temp->left==NULL && temp->right==NULL)
		return 1;                     //Returns 1 if node is a leaf node
	else
		return 0;
}

/**************************************************************************************************************
 Function that reports all the leaves that are accesible from the given node in left to right order
 
**************************************************************************************************************/
void node::reportsubtree(node *root)
{
 if(root==NULL)
 return ;
 else if(root->right ==NULL && root->left==NULL)
 cout<<root->fr<<" ";
 else
 {
  reportsubtree(root->left);
  reportsubtree(root->right);	
 }
}

