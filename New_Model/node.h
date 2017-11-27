#ifndef NODE_H
#define NODE_H

class node
{
 public:
 double num;
 long int fr;
 node *left;
 node *right;
 node *parent;
 int type;   
 int height;
 static node* newnode(double,long int);
 static int isLeaf(node *);
 static void reportsubtree(node *);
};

#endif

