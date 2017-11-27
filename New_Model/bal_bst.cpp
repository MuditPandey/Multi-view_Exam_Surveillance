#include "bal_bst.h"
#include"node.h"
#include "Misc_utils.h"
#include <bits/stdc++.h>

/************************************************************************************************************
 Function which dynamically inserts an element into the Binary Search Tree and maintains its balanced nature
 
************************************************************************************************************/

node* bal_bst::insert(node* node, double key,long int fr)
{
    /* 1.  Perform the normal BST insertion */
    if (node == NULL)
        return(node::newnode(key,fr));
 
    if (key < node->num)
        node->left  = insert(node->left, key,fr);
    else
        node->right = insert(node->right, key,fr);
 
    /* 2. Update height of this ancestor node */
    node->height = comp::max(height(node->left), height(node->right)) + 1;
 
    /* 3. Get the balance factor of this ancestor node to check whether
       this node became unbalanced */
    long long int balance = getBalance(node);
 
    // If this node becomes unbalanced, then there are 4 cases
 
    // Left Left Case
    if (balance > 1 && key < node->left->num)
        return rightRotate(node);
 
    // Right Right Case
    if (balance < -1 && key > node->right->num)
        return leftRotate(node);
 
    // Left Right Case
    if (balance > 1 && key > node->left->num)
    {
        node->left =  leftRotate(node->left);
        return rightRotate(node);
    }
 
    // Right Left Case
    if (balance < -1 && key < node->right->num)
    {
        node->right = rightRotate(node->right);
        return leftRotate(node);
    }
 
    /* return the (unchanged) node pointer */
    return node;
}

/************************************************************************************************************
 Function that returns the height of the node

************************************************************************************************************/
long long int bal_bst::height(struct node *N)
{
    if (N == NULL)
        return -1;
    return N->height;
}

/*************************************************************************************************************
 Function for left rotation of the AVL tree
 
*************************************************************************************************************/
node* bal_bst::leftRotate(struct node *x)
{
    node *y = x->right;
    node *T2 = y->left;
    // Perform rotation
    y->left = x;
    x->right = T2;
    //  Update heights
    x->height = comp::max(height(x->left), height(x->right))+1;
    y->height = comp::max(height(y->left), height(y->right))+1;
    // Return new root
    return y;
}
/*************************************************************************************************************
 Function for right rotation of the AVL tree
 
*************************************************************************************************************/
node* bal_bst::rightRotate(node *y)
{
    node *x = y->left;
    node *T2 = x->right;
    // Perform rotation
    x->right = y;
    y->left = T2;
    // Update heights
    y->height = comp::max(height(y->left), height(y->right))+1;
    x->height = comp::max(height(x->left), height(x->right))+1;
    // Return new root
    return x;
}

/************************************************************************************************************
 Function which returns the balance factor for the AVL tree

*************************************************************************************************************/

long long int bal_bst::getBalance(node *n)
{
    if (n== NULL)
        return 0;
    return height(n->left) - height(n->right);
}

/************************************************************************************************************
 Function to print the inorders traversal of the BST

*************************************************************************************************************/
void bal_bst::inorder(node *root)
{
 if(root==NULL)
 return;
 inorder(root->left);
 printf("%lf ",root->num);
 inorder(root->right);	
}



