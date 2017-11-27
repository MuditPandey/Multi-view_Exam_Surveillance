#ifndef BAL_BST_H
#define BAL_BST_H
#include "node.h"
#include<iostream>

class bal_bst
{
	public:
	static node* insert(node *root,double,long int);
	static void inorder(node *root);
	static long long int getBalance(node *);
    static node *rightRotate(node *);
    static node *leftRotate(node *);
    static long long int height(node *);
};

#endif

