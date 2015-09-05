title: "leetcode刷题小结"
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Leetcode刷题小结

## Array

### Median of two sorted array

There are two sorted arrays nums1 and nums2 of size m and n respectively. Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

更通用的形式为：给定两个已排序好的数组，找到两者所有元素中第k大的元素。

- 解法1：merge两个数组，然后求第k大的元素。O(m+n)复杂度。
- 解法2：利用一个计数器，记录当前已经找到的第m大的元素，从两个数组的第一个元素开始遍历。O(m+n)复杂度。
- 解法3：利用两个数组有序的特性，每次都删除k/2个元素。O(log(m+n))。

		class Solution {
		public:
		  // 寻找第k小的数
		  double find_kth(vector<int>::iterator it1, int n1,
						  vector<int>::iterator it2, int n2,
						  int k) {
			// 确保n1 >= n2
			if (n1 < n2) {
			  return find_kth(it2, n2, it1, n1, k);
			}
			if (n2 == 0) {
			  return *(it1 + k-1);
			}
			if (k == 1) {
			  return min(*it1, *it2);
			}
			// 注意这个划分,很重要
			int i2 = min(k/2, n2);
			int i1 = k - i2;
			if (*(it1 + i1-1) > *(it2 + i2-1)) {
			  // 删掉数组2的i2个
			  return find_kth(it1, n1, it2 + i2, n2 - i2, i1);
			} else if (*(it1 + i1-1) < *(it2 + i2-1)) {
			  // 删掉数组1的i1个
			  return find_kth(it1 + i1, n1 - i1, it2, n2, i2);
			} else {
			  return *(it1 + i1-1);
			}
		  }

		  // 寻找第k小的数, C语言版本
		  double find_kth2(const int* A, int m, const int* B, int n, int k) {
			if (m < n) {
			  return find_kth2(B, n, A, m, k);
			}
			if (n == 0) {
			  return A[k-1];
			}
			if (k == 1) {
			  return min(A[0], B[0]);
			}

			int i2 = min(k/2, n);
			int i1 = k - i2;
			if (A[i1-1] < B[i2-1]) {
			  return find_kth2(A+i1, m-i1, B, n, k-i1);
			} else if (A[i1-1] > B[i2-1]) {
			  return find_kth2(A, m, B+i2, n-i2, k-i2);
			} else {
			  return A[i1-1];
			}
		  }
		  // 数组从小到大排序
		  double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
			int total = nums1.size() + nums2.size();
			if (total & 0x1) {
			  // odd
			  // return find_kth(nums1.begin(), nums1.size(), nums2.begin(), nums2.size(), total/2 + 1);
			  return find_kth2(nums1.data(), nums1.size(), nums2.data(), nums2.size(), total/2 + 1);
			} else {
			  // return ( find_kth(nums1.begin(), nums1.size(), nums2.begin(), nums2.size(), total/2 + 1)
			  //     + find_kth(nums1.begin(), nums1.size(), nums2.begin(), nums2.size(), total/2) )/ 2.0;
			  return ( find_kth2(nums1.data(), nums1.size(), nums2.data(), nums2.size(), total/2 + 1)
				  + find_kth2(nums1.data(), nums1.size(), nums2.data(), nums2.size(), total/2) )/ 2.0;
			}

		  }
		};

### [H-Index II](https://leetcode.com/problems/h-index-ii/)
其实这个问题非常简单。写出来的话，主要是有一个需要注意的点，也就是二分查找。

1. right=n-1 => while(left  right=middle-1;
   right=n   => while(left  right=middle;

2. 注意middle的求法。(begin + end)/2 不对。(begin >> 1) + (end >> 1)也不对。begin + (end - begin) >> 1也不对。应该是begin + ((end - begin) >> 1)，注意这里犯过几次错了。
更多请参考[运算符优先级](http://baike.baidu.com/link?url=tS2RA_yjy-6sao8BQ1X-GafZJ9dhZaRy7kK1IQAxb1m5pDwXPE_Y_z1yRnVZbPLhD-NVbCTTzr2ZQCuJ2TRqAq)

    int hIndex(vector<int>& citations) {
        if (citations.empty()) {
            return 0;
        }
        int hindex = 0;
        int begin = 0;
        int end = citations.size() - 1;
        while (begin <= end) {
            int middle = begin + ((end - begin) >> 1);
            int lower = citations.size() - middle - 1;  // h of his/her N papers have at least h citations
            if (lower >= 0 && citations[lower] >= middle + 1) {
                if (middle + 1 > hindex) {
                    hindex = middle + 1;
                }
                begin = middle + 1;
            } else {
                end = middle - 1;
            }
        }
        return hindex;
    }

### [Contains Duplicate III](https://leetcode.com/problems/contains-duplicate-iii/)

Input: [-1,2147483647], 1, 2147483647

下面代码中，在计算gap时，首先gap必须是long类型，其次it_temp->first和last至少也有一个long，不然这个减法会有问题。

long gap = it_temp->first - last

除此外，还有一个容易犯的错误，gap的计算经常会在while循环里被忽视掉了。

主要可以参考 [隐式类型转换&& 负数的补码](http://www.cppblog.com/suiaiguo/archive/2009/07/16/90228.html)

	class Solution {
	public:
	    // from little to large
	    static bool SortFunction(const std::pair<int, int>& x, const std::pair<int, int>& y) {
	        if (x.first > y.first) {
	            return false;
	        } else if (x.first < y.first) {
	            return true;
	        } else {
	            return x.second < y.second;
	        }
	    }
	    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
	        if (nums.size() < 2) {
	            return false;
	        }
	        std::vector<std::pair<int, int> > middle;  // num -- index
	        for (int i = 0; i < nums.size(); ++i) {
	            middle.push_back(std::make_pair(nums[i], i));
	        }
	        std::sort(middle.begin(), middle.end(), SortFunction);
	        std::vector<std::pair<int, int> >::const_iterator it = middle.begin();
	        long last = it->first;
	        int index = it->second;
	        ++it;
	        for (; it != middle.end(); ++it) {
	            std::vector<std::pair<int, int> >::const_iterator it_temp = it;
	            long gap = it_temp->first - last;
	            while (it_temp != middle.end() && gap <= (long)t) {
	                // at most t && most k
	                if (abs(it_temp->second - index) <= k) {
	                    return true;
	                }
	                ++it_temp;
	                gap = it_temp->first - last;
	            }
	            last = it->first;
	            index = it->second;
	        }
	        return false;
	    }
	};


### [Kth Largest Element in an Array](https://leetcode.com/submissions/detail/30747333/)

    // 错误点: sort默认是从小到大排序
    void InsertArray(vector<int>& array, int insert) {
        // 二分查找
        int begin = 0;
        int end = array.size() - 1;
        int search_index = 0;
        while (begin <= end) {
            int middle = (long(begin) + long(end)) / 2;
            if (insert > array[middle]) {
                if (middle - 1 < 0 || insert < array[middle - 1]) {
                    search_index = middle;
                    break;
                }
                // 前半段
                end = middle;
            } else if (insert <= array[middle]) {
                if (middle + 1 > array.size() - 1 || insert >= array[middle + 1]) {
                    search_index = middle + 1;
                    break;
                }
                // 后半段
                begin = middle;
            }
        }
        // 找到index区间
        for (int i = array.size() - 1; i > search_index; --i) {
            array[i] = array[i-1];
        }
        array[search_index] = insert;
    }
    struct myclass {
        bool operator() (int i, int j) { return (i>j);}
    } myobject;

    int findKthLargest(vector<int>& nums, int k) {
        if (nums.size() < k || k < 1) {
            return 0;
        }
        vector<int> array(nums.begin(), nums.begin() + k);
        std::sort(array.begin(), array.end(), myobject);  // 从大到小排序
        for (int i = k; i < nums.size(); ++i) {
            if (nums[i] > array[k-1]) {
                InsertArray(array, nums[i]);
            } else {
                continue;
            }
        }
        return array[k - 1];
    }

### [堆排序]

[精通八大排序算法系列：二、堆排序算法](http://blog.csdn.net/v_july_v/article/details/6198644)

[堆排序 Wiki](https://zh.wikipedia.org/wiki/堆排序)

通常堆是通过一维数组(二叉树)来实现的。在起始数组为0的情形中：

    父节点i的左子节点在位置(2*i+1);
    父节点i的右子节点在位置(2*i+2);
    子节点i的父节点在位置floor((i-1)/2);

如果是最大堆，则父节点大于子节点。

那么在堆排序的时候，先通过调用(n-2)/2次Max_Heapify建立堆。
再移除位在第一个数据的根节点(这个最大堆的最大值)，并做最大堆(此时堆的size将减1)调整的递归运算。

    /*單一子結點最大堆積樹調整*/
    void Max_Heapify(int A[], int i, int heap_size)
    {
        int l = left(i);
        int r = right(i);
        int largest;
        int temp;
        if(l < heap_size && A[l] > A[i])
        {
            largest = l;
        }
        else
        {
            largest = i;
        }
        if(r < heap_size && A[r] > A[largest])
        {
            largest = r;
        }
        if(largest != i)
        {
            temp = A[i];
            A[i] = A[largest];
            A[largest] = temp;
            Max_Heapify(A, largest, heap_size);
        }
    }

    /*建立最大堆積樹*/
    void Build_Max_Heap(int A[],int heap_size)
    {
        for(int i = (heap_size-2)/2; i >= 0; i--)
        {
            Max_Heapify(A, i, heap_size);
        }
    }

    /*堆積排序程序碼*/
    void HeapSort(int A[], int heap_size)
    {
        Build_Max_Heap(A, heap_size);
        int temp;
        for(int i = heap_size - 1; i >= 0; i--)
        {
            temp = A[0];
            A[0] = A[i];
            A[i] = temp;
            Max_Heapify(A, 0, i);
        }
        print(A, heap_size);
    }

## List

### [Remove Linked List Elements](https://leetcode.com/problems/remove-linked-list-elements/)


    // 错误点: 未考虑都是val的情况. 也就是在unittest时，还是应该尽可能的考虑周全，要记得必须写unittest。
    // Input: [1,1], 1
    // 能否换一个思路,不再考虑删除,而是把不是val的node插入.
    // 本次错误的点:主要是没有考虑到连续的val存在. 对付这种题,可以先申明一个temp node;另外, 也就是在now->val == val的判断,对last的赋值要有一个else

    ListNode* removeElements(ListNode* head, int val) {
        if (head == NULL) {
            return head;
        }
        ListNode temp(val+1);
        temp.next = head;

        ListNode* last = &temp;
        ListNode* now = head;
        while (now) {
            if (now->val == val) {
                last->next = now->next;
            } else {
                last = now;
            }
            now = now->next;
        }
        return temp.next;
    }


## String

### [Isomorphic strings](https://leetcode.com/problems/isomorphic-strings/)

很简单的一个题目。但还是考虑不严谨。只是从s->t这个方面做了考虑，而没有考虑t->s这个方面。

    bool isIsomorphic(string s, string t) {
	    if (s.size() != t.size()) {
	        return false;
	    }
	    map<char, char> container1;
	    map<char, char> container2;
	    for (int i = 0; i < s.size(); ++i) {
	        map<char, char>::const_iterator it1 = container1.find(s[i]);
	        map<char, char>::const_iterator it2 = container2.find(t[i]);
	        if (it1 == container1.end()) {
	            container1[s[i]] = t[i];
	        } else {
	            if (it1->second != t[i]) {
	                return false;
	            }
	        }
	        if (it2 == container2.end()) {
	            container2[t[i]] = s[i];
	        } else {
	            if (it2->second != s[i]) {
	                return false;
	            }
	        }
	    }
	    return true;
    }

## 数据结构

### [Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/)

class Stack {
public:
    // Push element x onto stack.
    void push(int x) {
        in_.push_back(x);
        top_ = x;
    }

    // Removes the element on top of the stack.
    void pop() {
        if (in_.empty()) {
            return;
        } else if (in_.size() == 1) {
            in_.pop_front();
        } else {
            int in_size = in_.size();
            int i = 0;
            while (i < in_size - 1) {
                out_.push_back(in_.front());
                in_.pop_front();
                ++i;
            }
            in_.pop_front();
            while (!out_.empty()) {
                // in_.push_back(out_.front()); // 这里出错了.top_未赋值
                push(out_.front());
                out_.pop_front();
            }
        }
    }

    // Get the top element.
    int top() {
        return top_;
    }

    // Return whether the stack is empty.
    bool empty() {
        return in_.empty();
    }
private:
    deque<int> in_;
    deque<int> out_;
    int top_;
};


## 二叉树

### [Binary Tree Paths](https://leetcode.com/problems/binary-tree-paths/)

这里涉及到树的遍历，顺便把这些内容都回顾一下，主要参考一下这两篇文章

- [二叉树的深度优先遍历、广度优先遍历和非递归遍历](http://www.cnblogs.com/way_testlife/archive/2010/10/07/1845264.html)
- [二叉树的深度优先遍历与广度优先遍历-C++ 实现](http://www.blogjava.net/fancydeepin/archive/2013/02/03/395073.html)

**二叉树的深度优先遍历的非递归的通用做法是采用栈，广度优先遍历的非递归的通用做法是采用队列**

**非递归深度优先遍历二叉树**

栈是实现递归的最常用的结构，利用一个栈来记下尚待遍历的结点或子树，以备以后访问，可以将递归的深度优先遍历改为非递归的算法。

1. 非递归前序遍历：遇到一个结点，就访问该结点，并把此结点推入栈中，然后下降去遍历它的左子树。遍历完它的左子树后，从栈顶托出这个结点，并按照它的右链接指示的地址再去遍历该结点的右子树结构。

2. 非递归中序遍历：遇到一个结点，就把它推入栈中，并去遍历它的左子树。遍历完左子树后，从栈顶托出这个结点并访问之，然后按照它的右链接指示的地址再去遍历该结点的右子树。

3. 非递归后序遍历：遇到一个结点，把它推入栈中，遍历它的左子树。遍历结束后，还不能马上访问处于栈顶的该结点，而是要再按照它的右链接结构指示的地址去遍历该结点的右子树。遍历遍右子树后才能从栈顶托出该结点并访问之。另外，需要给栈中的每个元素加上一个特征位，以便当从栈顶托出一个结点时区别是从栈顶元素左边回来的(则要继续遍历右子树)，还是从右边回来的(该结点的左、右子树均已周游)。特征为Left表示已进入该结点的左子树，将从左边回来；特征为Right表示已进入该结点的右子树，将从右边回来。

4. 简洁的非递归前序遍历：遇到一个结点，就访问该结点，并把此结点的非空右结点推入栈中，然后下降去遍历它的左子树。遍历完左子树后，从栈顶托出一个结点，并按照它的右链接指示的地址再去遍历该结点的右子树结构。

    //深度优先遍历
    void depthFirstSearch(Tree root){
        stack<Node *> nodeStack;  //使用C++的STL标准模板库
        nodeStack.push(root);
        Node *node;
        while(!nodeStack.empty()){
            node = nodeStack.top();
            printf(format, node->data);  //遍历根结点
            nodeStack.pop();
            if(node->rchild){
                nodeStack.push(node->rchild);  //先将右子树压栈
            }
            if(node->lchild){
                nodeStack.push(node->lchild);  //再将左子树压栈
            }
        }
    }


**广度优先遍历二叉树**

广度优先周游二叉树(层序遍历)是用队列来实现的，从二叉树的第一层（根结点）开始，自上至下逐层遍历；在同一层中，按照从左到右的顺序对结点逐一访问。

按照从根结点至叶结点、从左子树至右子树的次序访问二叉树的结点。算法：

    1初始化一个队列，并把根结点入列队；

    2当队列为非空时，循环执行步骤3到步骤5，否则执行6；

    3出队列取得一个结点，访问该结点；

    4若该结点的左子树为非空，则将该结点的左子树入队列；

    5若该结点的右子树为非空，则将该结点的右子树入队列；

    6结束。

代码：

    //广度优先遍历
    void breadthFirstSearch(Tree root){
        queue<Node *> nodeQueue;  //使用C++的STL标准模板库
        nodeQueue.push(root);
        Node *node;
        while(!nodeQueue.empty()){
            node = nodeQueue.front();
            nodeQueue.pop();
            printf(format, node->data);
            if(node->lchild){
                nodeQueue.push(node->lchild);  //先将左子树入队
            }
            if(node->rchild){
                nodeQueue.push(node->rchild);  //再将右子树入队
            }
        }
    }

具体到原始问题的解答，代码如下。看起来还是蛮简单的，但是如果采用非递归的方法来做的话，还是蛮复杂的。
而且具体到递归，还有如何安排结果的输出，也是很有学问的。

    string ToString(int i) {
        stringstream ss;
        ss << i;
        return ss.str();
    }
    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> result;
        if (!root) {
            return result;
        }
        // 递归方法
        if (root->left == NULL && root->right == NULL) {
            result.push_back(ToString(root->val));
            return result;
        }
        if (root->left) {
            vector<string> result_left = binaryTreePaths(root->left);
            for (int i = 0; i < result_left.size(); ++i) {
                result.push_back(ToString(root->val) + "->" + result_left[i]);
            }
        }
        if (root->right) {
            vector<string> result_right = binaryTreePaths(root->right);
            for (int i = 0; i < result_right.size(); ++i) {
                result.push_back(ToString(root->val) + "->" + result_right[i]);
            }
        }
        return result;
    }

## 动态规划
更多动态规划的算法请参考 [geeksforgeeks DynamicProgramming](http://www.geeksforgeeks.org/fundamentals-of-algorithms/#DynamicProgramming)

### [Maximal Square](https://leetcode.com/problems/maximal-square/)

这里主要是利用动态规划来解，其方程为：

    动态规划：dp[x][y] = min(dp[x - 1][y - 1], dp[x][y - 1], dp[x - 1][y]) + 1
    上式中，dp[x][y]表示以坐标(x, y)为右下角元素的全1正方形矩阵的最大长度（宽度）

更多请参考 [largest-square-block](http://stackoverflow.com/questions/1726632/dynamic-programming-largest-square-block)

This is how the matrix will look like after the traversal. Values in parentheses are the counts, i.e. biggest square that can be made using the cell as top left.

    1(1) 0(0) 1(1) 0(0) 1(1) 0(0)
    1(1) 0(0) 1(4) 1(3) 1(2) 1(1)
    0(0) 1(1) 1(3) 1(3) 1(2) 1(1)
    0(0) 0(0) 1(2) 1(2) 1(2) 1(1)
    1(1) 1(1) 1(1) 1(1) 1(1) 1(1)

### [Ugly Number II](https://leetcode.com/problems/ugly-number-ii/)

[ugly-numbers answer](http://www.geeksforgeeks.org/ugly-numbers/)

每一个ugly numver都可以被2, 3, 5整除。
one way to look at the sequence is to split the sequence to three groups as below:

    (1) 1×2, 2×2, 3×2, 4×2, 5×2, 6*2, 8*2, 9*2 ...
    (2) 1×3, 2×3, 3×3, 4×3, 5×3, 6*3, 8*3, 9*3 ...
    (3) 1×5, 2×5, 3×5, 4×5, 5×5, 6*5, 8*5, 9*5 ...

We can find that every subsequence is the ugly-sequence itself (1, 2, 3, 4, 5, …) multiply 2, 3, 5. Then we use similar merge method as merge sort, to get every ugly number from the three subsequence. Every step we choose the smallest one, and move one step after.

## 其他

### [Rectangle Area](https://leetcode.com/problems/rectangle-area/)

    int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        int area = (C-A)*(D-B) + (G-E)*(H-F);
        if (A >= G || B >= H || C <= E || D <= F)
        {
            return area;
        }

        int top = min(D, H);
        int bottom = max(B, F);
        int left = max(A, E);
        int right = min(C, G);

        return area - (top-bottom)*(right-left);
    }

此题还可以换一种玩法，判断两个矩形是否相交。选择的方法可以是：求出top,bottom,left,right后，根据这四个点，判断是否可以组成一个矩形，也即验证是否有相交。

### [Largest Number](https://leetcode.com/problems/largest-number/)
该题目的关键就是定义：比较函数。思路是关键。前面绕了很多弯路。

    static bool compare(string &s1, string &s2)
    {
        return s1 + s2 > s2 + s1;
    }

    string largestNumber(vector<int> &num) {
        vector<string> arr;

        //将num转成string存入数组
        for(int i : num)
            arr.push_back(to_string(i));

        //比较排序
        sort(arr.begin(), arr.end(), compare);

        //连接成字符串
        string ret;
        for(string s : arr)
            ret += s;

        //排除特殊情况
        if(ret[0] == '0' && ret.size() > 0)
            return "0";

        return ret;
    }

### [Missing Number](https://leetcode.com/problems/missing-number/)

两个方法：求和，求异或。

    METHOD 1(Use sum formula)
    Algorithm:

    1. Get the sum of numbers
           total = n*(n+1)/2
    2  Subtract all the numbers from sum and
       you will get the missing number.

    METHOD 2(Use XOR)

      1) XOR all the array elements, let the result of XOR be X1.
      2) XOR all numbers from 1 to n, let XOR be X2.
      3) XOR of X1 and X2 gives the missing number.

### [First Missing Positive](https://leetcode.com/problems/first-missing-positive/)

相比于上一题，这道题要难一些，不能采用求和，求异或之类的方法来解了。只能再突破思路，想一想更多的点。

思路：交换数组元素，使得数组中第i位存放数值(i+1)。最后遍历数组，寻找第一个不符合此要求的元素，返回其下标。整个过程需要遍历两次数组，复杂度为O(n)。

### [Count primes](https://leetcode.com/problems/count-primes/)

思路很巧妙，关键还是算法。[Sieve_of_Eratosthenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes)

    int Label(int* array, int n, int p) {
        int multipler = 2;
        while (multipler * p < n) {
            array[multipler * p] = 1;
            multipler++;
        }
        for (int i = p+1; i < n; ++i) {
            if (array[i] == 0) {
                return i;
            }
        }
        return n;
    }
    int countPrimes(int n) {
        int* array = new int[n + 1];
        memset(array, 0, sizeof(int) * (n+1));
        int count = 0;
        for (int i = 2; i < n; ) {
            i = Label(array, n, i);
            count++;
        }
        delete[] array;
        return count;
    }

## Shell

### [Word Frequency](https://leetcode.com/problems/word-frequency/)

解答：

	awk -F" " '{for (i = 1; i <= NF; ++i) {num[$i]++;}}END{for (a in num) print a,num[a]|"sort -k2 -r -n"}' words.txt

注意几个细节：(1)在awk的输出中排序，可以在后面直接接sort命令，不过需要用引号。(2)这里是按照map的value排序，需要指定"-k2"。(3)注意是降序排列，所以有"-r"。(4)再注意默认是ascii排序，这里应该是number排序，所以有"-n"。

### [Transpose File ](https://leetcode.com/problems/transpose-file/)

有一个感触：awk内置的map如此强大。

	# (NF > p) {p = NF} 可以放到{}里面,如果在里面,则要加if.
	awk -F" " '{
	    for (i = 1; i <= NF; i++) {
	        content[NR,i] = $i
	    }

	}
	(NF > p) {p = NF}
	END{
	    for (i = 1; i <= p; i++) {
	        str = content[1, i]
	        for (j = 2; j <= NR; j++) {
	            str = str" "content[j, i]
	        }
	        print str
	    }
	}' file.txt

### [Valid Phone Numbers](https://leetcode.com/problems/valid-phone-numbers/)

这里主要考察正则表达式。具体tool可以使用：grep, egrep, sed, awk。

	#cat file.txt | grep -Eo '^(\([0-9]{3}\) ){1}[0-9]{3}-[0-9]{4}$|^([0-9]{3}-){2}[0-9]{4}$'
	#grep -Eo '^(\([0-9]{3}\) ){1}[0-9]{3}-[0-9]{4}$|^([0-9]{3}-){2}[0-9]{4}$' file.txt
	awk '/^(\([0-9]{3}\) ){1}[0-9]{3}-[0-9]{4}$|^([0-9]{3}-){2}[0-9]{4}$/' file.txt
	sed -n '/^(\([0-9]{3}\) ){1}[0-9]{3}-[0-9]{4}$/,/^([0-9]{3}-){2}[0-9]{4}$/p' file.txt

更多参考资料：

- [Sed简明教程-左耳朵耗子](http://coolshell.cn/articles/9104.html)
- [Awk简明教程-左耳朵耗子](http://coolshell.cn/articles/9070.html)
- [Sed by Example](http://www.funtoo.org/Sed_by_Example,_Part_2)
- [Regular Expressions](https://www.gnu.org/software/sed/manual/html_node/Regular-Expressions.html)
- [Awk regex](http://www.math.utah.edu/docs/info/gawk_5.html#SEC27)
