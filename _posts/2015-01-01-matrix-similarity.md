---
layout: post
title: "矩阵相似度计算"
description: ""
category:
tags: []
---

## 矩阵相似度计算

- 在机器学习任务中，矩阵是一个很重要的表述形式。文档与词，用户与其购买的商品，用户与好友关系等都可以描述成一个矩阵。为了描述方便，下文中矩阵都以U\*I代替，U代表user，I代表item，矩阵维数为m\*n。
对这个矩阵，一个最基础的任务就是找到最相似的用户或最相似的文档，也就是[k最近邻问题](http://zh.wikipedia.org/wiki/%E6%9C%80%E8%BF%91%E9%84%B0%E5%B1%85%E6%B3%95)(数据挖掘十大经典算法之一)。

#### 相似度计算方法
- 相似度计算方法：cosine距离，jaccard距离，bm25模型，proximity模型。具体请参考[机器学习中的相似性度量](http://www.cnblogs.com/heaad/archive/2011/03/08/1977733.html)

#### 降维方法
- 计算任意两个user之间的相似度，需要O(m*n)的复杂度。当n很大的时候，首先想到的办法是能否降维，将原矩阵变为m\*k维(k<<n)。

- 降维的方法有：svd，nmf，lsa，lda等。将一个大矩阵分解为两个小矩阵(m\*n分解为两个矩阵m\*k，k\*n)，或者分解为三个小矩阵(m\*n分解为两个矩阵m\*k，k\*k，k\*n)

#### minhash+lsh
- 除此之外，还有一种降维+局部敏感hash的算法。
也就是minhash + lsh。参考[MinHash wiki](http://en.wikipedia.org/wiki/MinHash)，[文本去重之MinHash算法](http://blog.csdn.net/sunlylorn/article/details/7835411)，[利用Minhash和LSH寻找相似的集合](http://www.cnblogs.com/bourneli/archive/2013/04/04/2999767.html)

	- 我们可以根据MinHash来计算两个集合的相似度了。一般有两种方法：

	- 第一种：使用多个hash函数。
为了计算集合A、B具有最小哈希值的概率，我们可以选择一定数量的hash函数，比如K个。然后用这K个hash函数分别对集合A、B求哈希值，对
每个集合都得到K个最小值。比如Min(A)k={a1,a2,...,ak}，Min(B)k={b1,b2,...,bk}。
那么，集合A、B的相似度为|Min(A)k ∩ Min(B)k| / |Min(A)k  ∪  Min(B)k|，及Min(A)k和Min(B)k中相同元素个数与总的元素个数的比例。

	- 第二种：使用单个hash函数。
第一种方法有一个很明显的缺陷，那就是计算复杂度高。使用单个hash函数是怎么解决这个问题的呢？请看：
前面我们定义过 hmin(S)为集合S中具有最小哈希值的一个元素，那么我们也可以定义hmink(S)为集合S中具有最小哈希值的K个元素。这样一来，
我们就只需要对每个集合求一次哈希，然后取最小的K个元素。计算两个集合A、B的相似度，就是集合A中最小的K个元素与集合B中最小的K个元素
的交集个数与并集个数的比例。

	- 对于每个user，利用minhash计算后，则将其从n维降维至K维向量。然后就该LSH出场了。

	- LSH:local sensitive hash。将上面K维向量划分到n个桶，每个桶有K/n维。两个user，只要有一个桶的元素是一样的，那么就认为他们是相似候选。这里有一个公式来衡量n的选值。请参考论文[find similar items](http://infolab.stanford.edu/~ullman/mmds/ch3.pdf)，[局部敏感哈希LSH科普](http://1.guzili.sinaapp.com/?p=190#more-190)

	![lsh](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/lsh.png)

#### 直接利用map-reduce
- 另外一种方法是不降维，直接通过map-reduce直接计算user之间的相似性。
	- 假设矩阵U\*I，计算两两U之间的相关性，时间复杂度是O(N^3)
	- 但是我们可以换个思路，将矩阵转置，以item为key。第一轮map-reduce过程，将U\*I矩阵转置为I\*U矩阵，输出每个item下，与该item有关联的所有user list。第二轮map，将同一个item下user两两组合成pair后输出，第二轮reduce，累加相同user pair的weight，得到任意两个user之间的相似度。
	- 具体请参考链接[大规模矩阵相似度计算](http://wbj0110.iteye.com/blog/2043700)。解决该问题就由两个Map-Reduce过程完成。第一个MR过程称为倒排索引，对每个文档，对其中的每个词语，以词语为键，文档标号与词语在该文档中的权重为值输出，这样，我们就得到如(F4,[(U1,0.1),(U2,0.9),(U7,0.5)])格式的输出。第二个MR过程计算文本相似度，以上一个MR过程的输出为输入，在Map过程中以文本对为键，以权重值的乘积为输出，比如上面的F4输出，map后变为[((U1,U2),0.09),((U1,U7),0.05),((U2,U7),0.45)]，这样，就得到了在所有的在两个文本中共同出现的词语针对该两个文本的权重乘积；然后在reduce过程中将相同键的值相加，就得到了所有的二元文本对的文本相似度。
	- 文中后面还讲了一些优化手段。

	![matrix_similairity](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/matrix_similairity.png)

####矩阵的乘法
- 额外再讲一点内容，矩阵的乘法,一个m\*k的矩阵A乘上一个k\*n的矩阵B，结果是一个m\*n的矩阵C。有两种分解方法：
	- 其一，把A矩阵按行分，把B矩阵按列分的观点来看矩阵乘法。C矩阵的一个子矩阵块可以看做是A对应多行和B对应多列的矩阵相乘得到的结果；
	- 其二，把矩阵A按列分块，矩阵B按行分块，A乘B可以等价于A的分块子矩阵乘上B中对应的分块子矩阵的加和。最特殊的情况是把A按列分为k个列向量，B按行分为k个行向量，然后对应的列向量于行向量相乘，得到k个矩阵，他们的和就是A和B的乘积。

- 矩阵乘法的并行方法
	-  第一种方法，按照A行B列的分解，我们将C矩阵分成m\*n个子矩阵，每个子矩阵由一个进程来计算。不考虑进程通讯问题，时间减少到单机版本的m\*n分之一。
	-  第二种方法，按照A列B行的分解，把C矩阵分解成k个同样大小的矩阵之和，分发到k个进程来计算，时间减少到单机版本的k分之一。
	-  哪一种方法更快，取决于k和m*n哪个更大。不过方法二要通信的数据量要明显大于方法一。
	-  哪一种方法需要存储更少，取决于(k+1)mn和(m+n)k的大小。

- 更多矩阵乘法，请参考文章[Cannon算法](http://en.wikipedia.org/wiki/Cannon's_algorithm)，[Scalable Universal Matrix Multiplication Algorithm](http://www.netlib.org/lapack/lawnspdf/lawn96.pdf)


#### 更多资料

- 昨天想用simhash做近似文档检测，感谢 @snowland @_aszxqw_ 等答疑。把这个过程中看的两篇文章和几篇介绍贴收集在这里 http://t.cn/RvlnyLl 这篇文章里 http://t.cn/RvlnyLW 作者论证MinHash更好。不过MinHash的Python实现好像没有simhash成熟。最后还是用了Liang Sun 的实现，自行实现持久化

- [机器学习日报-相似度计算](http://ml.memect.com/search/?q=simhash%20OR%20MinHash%20OR%20相似度)
