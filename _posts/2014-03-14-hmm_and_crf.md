---
layout: post
title: "HMM and CRF"
description: ""
category:
tags: [machine learning]
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

#HMM与CRF

#### author: vincentyao@tencent.com

看到语音识别的时候，觉得是该找个机会把HMM与CRF相关的知识点做一个总结了。
之前看过很多这方面的文章，但都是零零碎碎的，没有形成知识体系。

本文分为三部分，首先讲述生成模型与判别模型的定义与区别，接着分别阐述HMM和CRF的相关知识点，最后再讲述HMM与CRF的联系与区别，并讲述它们各自的应用点。

## 生成模型与判别模型

在讲HMM与CRF之前，先比较一下生成模型和判别模型。

### 生成模型，Generative Model

- 假设o是观察值，q是模型。如果对P(o\|q)建模，就是Generative模型。
- 其基本思想是首先建立样本的概率密度模型，再利用模型进行推理预测。一般建立在统计力学和bayes理论的基础之上。
- 估计的是联合概率分布（joint probability distribution），p(o, q)=p(o\|q)\*p(q)。
- 代表：Gaussians，Naive Bayes，HMMs，Bayesian networks，Markov random fields

### 判别模型，Discriminative Model

- 假设o是观察值，q是模型。如果对条件概率(后验概率) P(q\|o)建模，就是Discrminative模型。
- 基本思想是有限样本条件下建立判别函数，不考虑样本的产生模型，直接研究预测模型。代表性理论为统计学习理论。
- 估计的是条件概率分布(conditional distribution)， p(q\|o)。利用正负例和分类标签，focus在判别模型的边缘分布。目标函数直接对应于分类准确率。
- 代表：Logistic regression，SVMs，Neural networks，Conditional random fields(CRF)
- For instance, if y indicates whether an example is a dog (0) or an elephant (1), then p(x\|y = 0) models the distribution of dogs’ features, and p(x\|y = 1) models the distribution of elephants’ features.

### 模型对比

上面提到生成和判别模型，在具体讲述HMM与CRF之前，我们不妨先看一下各自的概率图，有一个形象直观的认识。

下图是HMM的概率图，属生成模型。以P(Y，X)建模，即P(o，q) = P(q)P(o\|q)建模。

![hmm1](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/hmm1.png)

下图是CRF的概率图，属判别模型。以P(Y\|X)建模。

![crf1](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/crf1.png)

## 隐马尔科夫模型

隐马尔科夫模型是由初始状态概率向量，状态转移概率矩阵，观测概率矩阵决定。

隐马尔科夫模型做了两个基本假设：

- 齐次马尔科夫性假设：假设隐藏的马尔科夫链在任意时刻t的状态只依赖于其前一个时刻，与其他时刻的状态和观测无关。
- 观测独立性假设：假设任意时刻的观测只依赖于该时刻的马尔科夫链的状态，与其他观测及状态无关。

三个基本问题：

- 概率计算问题。给定模型和观测系列，计算在模型下观测系列出现的概率。

	前向-后向算法。

- 学习问题。已知观测系列，估计模型参数，使得该模型下观测系列概率最大。

	EM算法，Baum-Welch算法。

- 预测问题，也称解码问题。已知模型和观测系列 O，求对给定观测系列，条件概率P(I\|O)最大的状态系列 I。

	Viterbi算法。

为什么是生成模型？

$$P(O|\lambda)=\sum_I P(O|I,\lambda)P(I|\lambda)$$

从上面公式可以看出，这是生成模型。
而观测系列的生成，与PLSA、LDA的生成过程类似。

## 条件随机域，CRF

### [视频: Log-linear Models and Conditional Random Fields](http://t.cn/SUGYtC)

Charles Elkan讲的对数线性模型和条件随机场，非常棒的教程。[讲义](http://t.cn/RZ1kQ6A)


### [Introduction to Conditional Random Fields](http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/)

中文分词目前学术上的state of art就是条件随机场搞的，场就是没有方向的，相互之间没有依赖关系，先后关系。而只有场的关系，能量关系。能量最小的“场面”是最趋向合理的。

以"Part-of-Speech Tagging"为示例。

在词性标注中，目标是：label a sentence (a sequence of words or tokens) with tags like ADJECTIVE, NOUN, PREPOSITION, VERB, ADVERB, ARTICLE。

举一个例子：给定一个句子“Bob drank coffee at Starbucks”, 词性标注的结果可能是：“Bob (NOUN) drank (VERB) coffee (NOUN) at (PREPOSITION) Starbucks (NOUN)”。

在CRF中, each feature function is a function that takes in as input:

- a sentence s
- the position i of a word in the sentence
- the label l_i of the current word
- the label l_i−1 of the previous word

and outputs a real-valued number (though the numbers are often just either 0 or 1).

(备注:building the special case of a linear-chain CRF)

Next, assign each feature function fj a weight λj

Given a sentence s, we can now score a labeling l of s by adding up the weighted features over all words in the sentence:

$$score(l\|s) = \sum_{j=1}^m \sum_{i=1}^n {\lamda_j f_j(s,i,l_i,l_{i-1})}$$

(The first sum runs over each feature function j, and the inner sum runs over each position i of the sentence.)

Finally, we can transform these scores into probabilities p(l|s) between 0 and 1 by exponentiating and normalizing:

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/crf_ probabilities.png)

**Smells like Logistic Regression**

That’s because CRFs are indeed basically the sequential version of logistic regression: whereas logistic regression is a log-linear model for classification, CRFs are a log-linear model for sequential labels.

**Looks like HMMs**

Recall that Hidden Markov Models are another model for part-of-speech tagging (and sequential labeling in general). Whereas CRFs throw any bunch of functions together to get a label score, HMMs take a generative approach to labeling, defining

$$p(l,s)=p(l_1)\prod_i {p(l_i|l_{i−1})p(w_i|l_i)}$$

where

p(li\|li−1) are transition probabilities (e.g., the probability that a preposition is followed by a noun);
p(wi\|li) are emission probabilities (e.g., the probability that a noun emits the word “dad”).

So how do HMMs compare to CRFs? CRFs are more powerful – they can model everything HMMs can and more. One way of seeing this is as follows.

## 模型之间的联系
从下面两张图看各个模型之间的联系：

![crf_hmm1](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/crf_hmm1.png)

![crf_hmm2](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/crf_hmm2.png)

- Naive bayes和HMM属于生成模型，因为他们估计的是联合分布。
- MaxEnt和CRF属于判别模型，因为他们估计的是条件概率分布。
- NB和ME中，输出Y只有一个class。
- HMM和CRF中，输出Y有多个，是sequence，属于structured prediction。

## 应用之"中文分词"

主要用于基于字标注的分词。例如 “我喜欢天安门” 就可以变成这样的标注 “我s喜b欢e天b安m门e”。
通过s（single）b（begin）m（middle）e（end）这样的标注把分词问题转变为标注问题。

**HMM**

HMM隐藏马尔可夫链模型就是这样一个字标注的分词算法，假设原来的句子序列是\\(a_1,a_2,a_3,...,a_n\\)，标注序列是\\(c_1,c_2,...,c_n\\)，那么HMM是要求这样的式子:

$$argmax \prod{P(c_i|c_{i-1})*P(a_i|c_i)}$$

**最大熵模型ME（Maximum Entropy）**

最大熵模型一般就是在已知条件下，来求熵最大的情况，最大熵模型我们一般会有feature函数，给定的条件就是样本期望等于模型期望，即:

$$\overline{p}(f)=\Sigma{\overline{p}(a_i,c_i)*f(a_i,c_i)}=p(f)=\Sigma{p(c_i|a_i)*\overline{p}(a_i)*f(a_i,c_i)}$$

在已知条件下，求熵最大的情况

$$argmax H(c_i|a_i)$$

H就是信息熵的函数，于是这样我们就求出了\\(P(c_i\|a_i)\\)，就知道了每个字a的标注c了，最大熵模型的一个好处是我们可以引入各种各样的feature，而不仅仅是从字出现的频率去分词，比如我们可以加入domain knowledge，可以加入已知的字典信息等。

**最大熵马尔可夫模型MEMM（Maximum-entropy Markov model）**

最大熵模型的一个问题就是把每个字的标注问题分裂来看了，于是就有人把马尔可夫链和最大熵结合，搞出了最大熵马尔可夫模型，这样不仅可以利用最大熵的各种feature的特性，而且加入了序列化的信息，使得能够从整个序列最大化的角度来处理，而不是单独的处理每个字，于是MEMM是求这样的形式：

$$argmax\prod{P(c_i|c_{i-1},a_i)}$$

**条件随机场CRF（Conditional Random Field）**

MEMM的不足之处就是马尔可夫链的不足之处，马尔可夫链的假设是每个状态只与他前面的状态有关，这样的假设显然是有偏差的，所以就有了CRF模型，使得每个状态不止与他前面的状态有关，还与他后面的状态有关。HMM是基于贝叶斯网络的有向图，而CRF是无向图。

$$P(Y_v|Y_w,w \neq v)=P(Y_v,Y_w,w \sim v)$$
where w~v means that w and v are neighbors in G.

上式是条件随机场的定义，一个图被称为条件随机场，是说图中的结点只和他相邻的结点有关。最后由于不是贝叶斯网络的有向图，所以CRF利用团的概念来求，最后公式如下

$$P(y|x,\lambda)=\frac{1}{Z(x)}*exp(\Sigma{\lambda_j*F_j(y,x)})$$

因为条件随机场既可以像最大熵模型那样加各种feature，又没有马尔可夫链那样的偏执假设， 所以近年来CRF已知是被公认的最好的分词算法[StanfordNLP](http://nlp.stanford.edu/software/segmenter.shtml)里就有良好的中文分词的CRF实现，在他们的[论文 Optimizing Chinese Word Segmentation for Machine Translation Performance](http://nlp.stanford.edu/pubs/acl-wmt08-cws.pdf)提到，他们把字典作为feature加入到CRF中，可以很好的提高分词的performance。

## 应用之"命名实体识别"

## 参考文献
- [classical probabilistic model and conditional random field](http://www.scai.fraunhofer.de/fileadmin/images/bio/data_mining/paper/crf_klinger_tomanek.pdf)
- [An Introduction to Conditional Random Fields for Relational Learning](http://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf)
- [隐马尔可夫模型 最大熵马尔可夫模型 条件随机场 区别和联系](http://1.guzili.sinaapp.com/?p=133#comment-151)  该文章待会好好读下
- [52nlp hmm](http://www.52nlp.cn/tag/hmm)
- [浅谈中文分词](http://www.isnowfy.com/introduction-to-chinese-segmentation/)
- [机器学习“判定模型”和“生成模型”有什么区别？](http://www.zhihu.com/question/20446337)

CRF:

- [An Introduction to Conditional Random Fields. by Charles Sutton](http://arxiv.org/pdf/1011.4088v1.pdf)
- Conditional Random Fields as Recurrent Neural Networks [link](http://t.cn/Rwbbmq1) 就喜欢这种把model串起来的工作方便理解和泛化。paper将mean-field inference每次迭代过程和CNN对应上，整个inference过程对应为一个Recurrent NN 这是这几天arxiv中算有意思的paper
- [How conditional random fields are ‘powerful’ in machine learning - Techworld](http://t.cn/R7D3BbE)
- 1)#数据挖掘十大算法#是香港ICDM06年从18个候选中投票产生；候选由KDD创新奖和ICDM研究贡献奖得主各自可提名十个、然后经谷歌学术删除掉引用少于50而得之 http://t.cn/zOIpSia 2)快十年过去了；Netflix搞推荐系统的Amatriain提出自己的Top10：MF GBDT RF ANN LR CRF LDA http://t.cn/RZ8kGW9
- 用MeCab打造一套实用的中文分词系统: MeCab是一套优秀的日文分词和词性标注系统,基于CRF打造,有着诸多优点,代码基于C++实现，基本内嵌CRF++代码，性能优良，并通过SWIG提供多种语言调用接口, 可扩展性和通用性都非常不错。这篇[博客](http://t.cn/RZjgtM0)尝试基于MeCab训练一套中文分词系统，欢迎观摩。
- [CRF++学习](http://blog.csdn.net/gududanxing/article/details/10827085)
- [三种CRF实现在中文分词任务上的表现比较](https://jianqiangma.wordpress.com/2011/11/14/%E4%B8%89%E7%A7%8Dcrf%E5%AE%9E%E7%8E%B0%E7%9A%84%E7%AE%80%E5%8D%95%E6%AF%94%E8%BE%83/)
- [CRF++ library](http://crfpp.googlecode.com/svn/trunk/doc/index.html?source=navbar)
- CRF训练，但标注数据很少。可以参考：Semi-supervised Sequence Labeling for Named Entity Extraction based on Tri-Training:Case Study on Chinese Person Name Extraction
- 推荐这个[项目](http://leon.bottou.org/projects/sgd)，虽然现在都流行 Deep Learning了，CRF 类方法还是很容易达到一个比较高的 Score，这个项目f-score 低了 0.7%，但是速度提升了10倍，隐含的，可以处理更大量的样本数据。
- 机器学习班第15次课，邹博讲条件随机场CRF的PPT [下载地址](http://t.cn/RzE4Oy8)，第16次课，邹博讲PCA&SVD的PPT [下载地址](http://t.cn/RzE4OyQ)，@sumnous_t 讲社区发现算法的PPT [下载地址](http://t.cn/RzE4OyR)。
