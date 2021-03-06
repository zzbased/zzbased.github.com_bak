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



## 动态规划

### [Maximal Square](https://leetcode.com/problems/maximal-square/)

这里主要是利用动态规划来解，其方程为：

         动态规划：dp[x][y] = min(dp[x - 1][y - 1], dp[x][y - 1], dp[x - 1][y]) + 1
         上式中，dp[x][y]表示以坐标(x, y)为右下角元素的全1正方形矩阵的最大长度（宽度）

更多请参考 [largest-square-block](http://stackoverflow.com/questions/1726632/dynamic-programming-largest-square-block)

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
        /*
        for (int i = 2; i < n; ++i) {
            if (array[i] == 0) {
                count++;
            }
        }*/
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
