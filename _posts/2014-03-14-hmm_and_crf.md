---
layout: post
title: "HMM and CRF"
description: ""
category:
tags: [machine learning]
---


<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

##HMM与CRF

看到语音识别的时候，觉得是该找个机会把HMM与CRF相关的知识点做一个总结了。
之前看过很多这方面的文章，但都是零零碎碎的，没有形成知识体系。

### 推荐文章

首先推荐几篇文章：

[classical probabilistic model and conditional random field](http://www.scai.fraunhofer.de/fileadmin/images/bio/data_mining/paper/crf_klinger_tomanek.pdf)

[An Introduction to Conditional Random Fields for Relational Learning](http://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf)

[隐马尔可夫模型 最大熵马尔可夫模型 条件随机场 区别和联系](http://1.guzili.sinaapp.com/?p=133#comment-151)  该文章待会好好读下

[52nlp hmm](http://www.52nlp.cn/tag/hmm)

[浅谈中文分词](http://www.isnowfy.com/introduction-to-chinese-segmentation/)

### 模型之间的联系
从下面两张图看各个模型之间的联系：

![crf_hmm1](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/crf_hmm1.png)

![crf_hmm2](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/crf_hmm2.png)

- Naive bayes和HMM属于生成模型，因为他们估计的是联合分布。
- MaxEnt和CRF属于判别模型，因为他们估计的是条件概率分布。
- NB和ME中，输出Y只有一个class。
- HMM和CRF中，输出Y有多个，是sequence。

### 生成模型与判别模型

#### 生成模型，Generative Model

- 假设o是观察值，q是模型。如果对P(o\|q)建模，就是Generative模型。
- 其基本思想是首先建立样本的概率密度模型，再利用模型进行推理预测。一般建立在统计力学和bayes理论的基础之上。
- 估计的是联合概率分布（joint probability distribution），p(class, context)=p(class\|context)\*p(context)。
- 代表：Gaussians，Naive Bayes，HMMs，Bayesian networks，Markov random fields

#### 判别模型，Discriminative Model

- 假设o是观察值，q是模型。如果对条件概率(后验概率) P(q\|o)建模，就是Discrminative模型。
- 基本思想是有限样本条件下建立判别函数，不考虑样本的产生模型，直接研究预测模型。代表性理论为统计学习理论。
- 估计的是条件概率分布(conditional distribution)， p(class\|context)。利用正负例和分类标签，focus在判别模型的边缘分布。目标函数直接对应于分类准确率。
- 代表：logistic regression，SVMs，neural networks，Conditional random fields(CRF)

### 隐马尔科夫模型

隐马尔科夫模型是由初始状态概率向量，状态转移概率矩阵，观测概率矩阵决定。

隐马尔科夫模型做了两个基本假设：

- 齐次马尔科夫性假设：假设隐藏的马尔科夫链在任意时刻t的状态只依赖于其前一个时刻，与其他时刻的状态和观测无关。
- 观测独立性假设：假设任意时刻的观测只依赖于该时刻的马尔科夫链的状态，与其他观测及状态无关。

三个基本问题：

- 概率计算问题。给定模型和观测系列，计算在模型下观测系列出现的概率。
  前向-后向算法。
- 学习问题。已知观测系列，估计模型参数，使得该模型下观测系列概率最大。
  EM算法，Baum-Welch算法。
- 预测问题，也称解码问题。已知模型和观测系列，求对给定观测系列条件概率P(I|O)最大的状态系列。
  Viterbi算法。

为什么是生成模型？

$$P(O|\lambda)=\sum_I P(O|I,\lambda)P(I|\lambda)$$

从上面公式可以看出，这是生成模型。
而观测系列的生成，与LDA的生成过程类似。

### 条件随机域，CRF
- [CRF++学习](http://blog.csdn.net/gududanxing/article/details/10827085)
- [三种CRF实现在中文分词任务上的表现比较](https://jianqiangma.wordpress.com/2011/11/14/%E4%B8%89%E7%A7%8Dcrf%E5%AE%9E%E7%8E%B0%E7%9A%84%E7%AE%80%E5%8D%95%E6%AF%94%E8%BE%83/)
- [CRF++ library](http://crfpp.googlecode.com/svn/trunk/doc/index.html?source=navbar)
- CRF训练，但标注数据很少。感兴趣的朋友可以参考下Semi-supervised Sequence Labeling for Named Entity Extraction based on Tri-Training:Case Study on Chinese Person Name Extraction
- [视频]《Log-linear Models and Conditional Random Fields》http://t.cn/SUGYtC Charles Elkan讲的对数线性模型和条件随机场，非常棒的教程 讲义:http://t.cn/RZ1kQ6A
- http://t.cn/zO7uh30 推荐这个项目，虽然现在都流行 Deep Learning了， CRF 类方法还是很容易达到一个比较高的 Score， 这个项目 f-score 低了 0.7 % 但是速度 提升了 10倍，隐含的，可以处理更大量的样本数据。
- PPT 来了！机器学习班第15次课，邹博讲条件随机场CRF的PPT 下载地址：http://t.cn/RzE4Oy8，第16次课，邹博讲PCA&SVD的PPT 下载地址：http://t.cn/RzE4OyQ，@sumnous_t 讲社区发现算法的PPT 下载地址：http://t.cn/RzE4OyR。顺便说句，sumnous还曾是算法班周六班的学员，一年下来，进步很大。分享！
- [文章]《Introduction to Conditional Random Fields》(2012) http://t.cn/S67yJs 很好的条件随机场(CRF)介绍文章（学习笔记）

  中文分词目前学术上的state of art就是条件随机场搞的，场就是没有方向的，相互之间没有依赖关系，先后关系。而只有场的关系，能量关系。能量最小的“场面”是最趋向合理的。

- [Introduction to Conditional Random Fields](http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/)

	这篇文章讲得很好，有必要在明天早上把这篇文章再好好完善一下。然后把CRF与HMM整理出一片文章来。

- Conditional Random Fields as Recurrent Neural Networks link：http://t.cn/Rwbbmq1 就喜欢这种把model串起来的工作方便理解和泛化。paper将mean-field inference每次迭代过程和CNN对应上，整个inference过程对应为一个Recurrent NN 这是这几天arxiv中算有意思的paper@火光摇曳Flickering

- How conditional random fields are ‘powerful’ in machine learning - Techworld http://t.cn/R7D3BbE

- 1)#数据挖掘十大算法#是香港ICDM06年从18个候选中投票产生；候选由KDD创新奖和ICDM研究贡献奖得主各自可提名十个、然后经谷歌学术删除掉引用少于50而得之 http://t.cn/zOIpSia 2)快十年过去了；Netflix搞推荐系统的Amatriain提出自己的Top10：MF GBDT RF ANN LR CRF LDA http://t.cn/RZ8kGW9

- 用MeCab打造一套实用的中文分词系统: MeCab是一套优秀的日文分词和词性标注系统,基于CRF打造,有着诸多优点,代码基于C++实现，基本内嵌CRF++代码，性能优良，并通过SWIG提供多种语言调用接口, 可扩展性和通用性都非常不错。这篇博客尝试基于MeCab训练一套中文分词系统，欢迎观摩 http://t.cn/RZjgtM0


### 对比
![hmm1](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/hmm1.png)

上图是HMM的概率图，属生成模型。以P(Y,X)建模，即P(O，q) = P(q)P(O\|q) 建模。

![crf1](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/crf1.png)

上图是CRF的概率图，属判别模型。以P(Y\|X)建模。

### 参考文献
- [Markdown中插入数学公式的方法](http://blog.csdn.net/xiahouzuoxin/article/details/26478179)
- [LaTeX/数学公式](http://zh.wikibooks.org/zh-cn/LaTeX/%E6%95%B0%E5%AD%A6%E5%85%AC%E5%BC%8F)
- [LaTeX数学公式输入初级](http://blog.sina.com.cn/s/blog_5e16f1770100fs38.html)



