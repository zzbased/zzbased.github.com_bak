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


## Shell

### [Word Frequency](https://leetcode.com/problems/word-frequency/)

解答：

	awk -F" " '{for (i = 1; i <= NF; ++i) {num[$i]++;}}END{for (a in num) print a,num[a]|"sort -k2 -r -n"}' words.txt

注意几个细节：(1)在awk的输出中排序，可以在后面直接接sort命令，不过需要用引号。(2)这里是按照map的value排序，需要指定"-k2"。(3)注意是降序排列，所以有"-r"。(4)再注意默认是ascii排序，这里应该是number排序，所以有"-n"。

### [Transpose File ](https://leetcode.com/problems/transpose-file/)

有一个细节：awk内置的map如此强大。

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