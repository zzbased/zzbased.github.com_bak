---
layout: post
title: "推荐算法总结"
description: ""
category:
tags: [machine learning]
---
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


# 推荐算法总结

#### author: vincentyao@tencent.com

互联网的发展经历了这样几个阶段，首先是信息产生阶段，接着由于信息越来越多，用户需要查询相关信息，就有了搜索引擎，再接着，信息过载更加严重，用户需求不明确时，就需要推荐引擎了。

推荐在计算广告上有很多的运用，如下图所示，Netflix有2/3的电影观看都是经由推荐的。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/value_of_recommender.png)

这里计划按照自己的理解对推荐算法做一个总结。下面内容主要分为三节，第一节介绍各种推荐算法，包括传统算法(CF)，新算法(Deeplearning)等；第二节精选了业界的一些成熟推荐系统，做一下简要的学习与总结；第三节则是自己对推荐系统的理解与总结。

## 推荐算法介绍

### 推荐问题定义
Estimate a utility function that automatically predicts how a user will like an item。

### Memory-based Collaborative Filtering
- 协同（collaborating）是群体行为，过滤（filtering）则是针对个人的行为。
- 假设有m个user，n个item。每个用户都有一个关联的item list，并且通过显式或隐式的反馈，对每个item都有一个rating \\(v_{i,j}\\)。
- 显式评分指用户使用系统提供的方式进行评分或者评价; 隐式评分则根据使用者的行为模式由系统代替使用者完成评价，行为模式包括用户的浏览行为、购买行为等等。
- 基本步骤为：(1)收集用户评分；(2)最近邻搜索；(3)产生推荐结果。第3步中，较常见的推荐算法有Top-N推荐和关联推荐。Top-N推荐比较熟悉，关联推荐是对最近邻使用者的记录进行关联规则(association rules)挖掘。

- 长处：
	- minimal knowledge，content-agnostic，可以在不理解item和user的情况下，做任何类型的推荐。
- 短处：
	- 需要大量可信的用户反馈数据
	- 不考虑上下文信息
	- 商品都是标准化的(Users should have bought exactly the same product)
	- content-agnostic（与内容无关的），容易推荐popular items，即Popularity Bias。
	- new and unpopular items cannot be recommended，即cold-start problem。

#### User-based
先找最近邻user，再基于最近邻user预测。

- KNN：一个基于Python的Nearest Neighbors Search库 http://t.cn/RvVr8Gb。博文介绍 http://t.cn/RwSZk2l。另外基于这个库的一个推荐系统 http://t.cn/RZ339rQ，将作为一个demo presentation在WWW上出现。作者是剑桥的博士后 @唧唧歪歪de计算机博士

- User之间相似度计算
	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/user_based_similarity.png)

- Prediction for user i and items j：
	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/user_based_prediction.png)

#### Item-based
先计算item similarity，再基于user rated items预测。

- 考虑到不同的用户，对于不同的物品，都有不同的打分偏置；所以需要做归一化与偏置矫正。rate bias：\\(b_{ui} = μ(global) + b_u(user bias) + b_i(item bias)\\)，\\(s_k(i,u)\\) is k-nearest neighbors to i that were rated by user u。

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/cf_formula.png)
	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/user_item_bias.png)

- item-based计算流程：

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images//item_similarity.png)

#### Association rules(关联规则)
- 关联规则分析 (Association Rules，又称 Basket Analysis) 用于从大量数据中挖掘出有价值的数据项之间的相关关系。经典论文[Mining Association Rules between Sets of Items in Large Databases]()。关联规则解决的常见问题如：“如果一个消费者购买了产品A，那么他有多大机会购买产品B?”以及“如果他购买了产品C和D，那么他还将购买什么产品？”

- 常见算法有：Apriori算法和FP-growth算法。请参考[wiki](http://zh.wikipedia.org/wiki/先验算法)，[Frequent ItemSets : Apriori Algorithm and Example](http://t.cn/RwCAsfM)，[link2](http://t.cn/RwC2vEY)

- 关联规则面向的是transaction，而User-based or Item based面向的是用户偏好（评分），协同过滤在计算相似商品的过程中可以使用关联规则分析。具体请参考[协同过滤和关联规则分析的区别是什么](http://www.zhihu.com/question/22404652)

#### Memory-based CF总结
- Problem: sparsity，scalability
	- 针对sparsity，可以利用latent models做降维。Methods of dimensionality reduction: Matrix Factorization, Clustering，Projection(PCA) ...
	- scalability瓶颈点：相关性计算。将最近邻产生与预测分为两个步骤，其中相关性计算时间复杂度很高。

- 个人总结，一般来讲，item-based方法更好用，因为item之间的similarity是相对静态，但user之间的similarity相对动态。当item base is smaller than user or changes rapidly时，采用user-based方法更合适；相反，当user base is small时，Item-based方法更合适；

### Model-based Collaborative Filtering

#### SVD/MF
- SVD
	- U是user-factor矩阵，V是item-factor矩阵。
		![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/model_based_svd.png)

	- 基于svd的rating过程：
		![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/svd_rating.png)

	- 先写出loss function，再利用SGD or Alternating least squares求解
		![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/svd_object_function.png)

		增加bias后的目标优化函数(loss function)：
		![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/svd_object_function_bias.png)

		关于user/item biases的作用，补充几点：(1)偏好信息的充分利用；(2)能充分利用用户、物品的profile等属性信息；(3)属性之间能方便的进行各种组合。

		关于求解算法，Alternating least squares方法的原理是：首先固定item vectors，最优化user vectors，再固定user vectors，最优化item vectors。SGD就是梯度下降法，相对更快。

	- 算法优点：(1)将用户和物品用隐特征(latentfeature)连接在一起；(2)MatrixFactorization有明确的数学理论基础(singularvalue)和优 化目标,容易逼近最优解；(3)对数据稀疏性(datasparsity)和抗噪音干扰的处理效果较好; (4)延展性(scalability)很好;
	- 算法缺点：(1)可解释性弱；(2)难以实时更新(适用于离线计算)；(3)Overfitting without regularization，特别是fewer reviews than dimensions。

- SVD++

	SVD++ 是SVD模型的加强版，除了打分关系，SVD++还可以对隐含的回馈(implicit feedback) 进行建模。
	除了在SVD中定义的向量外，每个item对应一个向量yi，来通过user隐含回馈过的item的集合来刻画用户的偏好。

	$$\hat{r}_{ui} = \mu + b_i + b_u + q_i^T (p_u + |R(u)|^{-\frac{1}{2}} \sum_{j\in R(u)} y_i)$$

	其中， R(u) 代表user隐含回馈（打分过的）过的item的集合。
	可以看到，现在user被建模为\\( p_u + |R(u)|^{-\frac{1}{2}} \sum_{j\in R(u)} y_i \\)，

	具体请参考[文章SVD/SVD++](http://www.superchun.com/machine-learning/svd1.html)，[论文Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model](http://research.yahoo.com/files/kdd08koren.pdf)

- SVDfeature

	[SVDFeature: A Toolkit for Feature-based Collaborative Filtering](http://www.jmlr.org/papers/volume13/chen12a/chen12a.pdf)

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/svdfeature.png)

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/svdfeature2.png)

- 开源库，请参考[推荐系统开源软件列表汇总和点评](http://blog.csdn.net/cserchen/article/details/14231153)
	- [libFM](http://www.libfm.org)，by Steffen Rendle。特点是实现了MCMC（Markov Chain Monte Carlo）优化算法，比常见的SGD（随即梯度下降）优化方法精度要高（当然也会慢一些）。
	- [svdfeature](http://svdfeature.apexlab.org/wiki/Main_Page)，by 上海交大。
	- [libMF](http://www.csie.ntu.edu.tw/~cjlin/libmf/)，by国立台湾大学。参考论文[Y. Zhuang, W.-S. Chin, Y.-C. Juan, and C.-J. Lin. A Fast Parallel SGD for Matrix Factorization in Shared Memory Systems](http://www.csie.ntu.edu.tw/~cjlin/papers/libmf/libmf_journal.pdf)。
	- [NMF](http://www.csie.ntu.edu.tw/~cjlin/nmf/)

#### RBM

[论文 Restricted Boltzmann Machines for Collaborative Filtering]()

#### Clustering

将用户聚类后，再基于传统CF的方法(此时user不是单独的用户，而是cluster)

#### Locality-sensitive hashing

- Method for grouping similar items in highly dimensional spaces；
- Find a hashing function s.t. similar items are grouped in the same buckets;
- 可以参考站内文章[矩阵相似度计算](http://zzbased.github.io/2015/01/01/matrix-similarity.html)

#### Classifiers
Classifiers can be used in CF and CB Recommenders。
优点：可以和其他方法结合使用。缺点是：需要一份训练集。

### Content-based Recommenders

这里可以利用众多自然语言处理技术，分别建立用户/Item画像(category, tag/keyword，topic等)，然后基于两者画像做相关性计算。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/content_based.png)

### New approaches

#### Learning to Rank
- 一个机器学习问题，目标是从训练数据里构建一个ranking model。排序学习指在排序生成 (ranking creation) 和排序整合 (ranking aggregation) 中用于构建排序模型的机器学习方法。[slice](http://www.icst.pku.edu.cn/lcwm/course/WebDataMining/slides2012/8机器学习及排序学习基础.pdf)，[Learning to Rank简介](http://www.cnblogs.com/kemaswill/archive/2013/06/01/3109497.html)
- Learning to rank is a key element for personalization
- Treat the problem as a standard supervised classification problem
- [Ranklib code](http://people.cs.umass.edu/~vdang/ranklib.html)，[svmlight](http://svmlight.joachims.org)

- Pointwise
	- Ranking score based on regression or classification
	- LR, SVM, GBDT, McRank ...
- Pairwise
	- Randing problem是二分类问题，pair-wise的比较，将排序问题转换为分类问题
	- RankSVM, RankBoost, RankNet, FRank ...
- Listwise
	- ListNet: KL-divergence as loss function by define a probability distribution
	- RankCosine: similarity between ranking list and ground truth as loss function
	- Lambda Rank，ListNet，ListMLE，AdaRank，SVMap ...

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/learning_to_rank_compare.png)

- [关于learning to rank的讨论](http://hao.memect.com/?tag=learningtorank+weibo)，[learning to rank的推荐论文，25篇](http://hao.memect.com/?tag=learningtorank+pdf)


#### Context-aware Recommendations
- [论文Context-Aware Recommender Systems](http://ids.csom.umn.edu/faculty/gedas/NSFCareer/CARS-chapter-2010.pdf)

- 从[R: User * Item -> Rating] 到 [R: User * Item * Context -> Rating]

- 传统推荐过程框图
	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/general-recommender.png)

- 将context纳入推荐系统后
	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/context-recommender1.png)
	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/context-recommender2.png)

- 两种方法：
	- Tensor Factorization
		![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/hosvd.png)
	- Factorization Machines

#### Deep learning: ANN training, Recurrent Networks
- [Recurrent Neural Networks for Collaborative Filtering](http://erikbern.com/?p=589)，[Collaborative Filtering at Spotify](http://www.slideshare.net/erikbern/collaborative-filtering-at-spotify-16182818?related=1)
Recurrent neural networks have a simple model that tries to predict the next item given all previous ones。

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/cf-RNN-1.png)

- [Recommending music on Spotify with deep learning](http://benanne.github.io/2014/08/05/spotify-cnns.html)

#### Similarity: graph-based similarity(simrank)

- Graph-based similarities
	- SimRank: two objects are similar if they are referenced by similar objects
	- [论文 SimRank](http://www-cs-students.stanford.edu/~glenj/simrank.pdf)
	- [SimRank原理](http://www.spiral.pro/big_data/simrank-intro.html)

#### Social Recommendations
- Social and Trust-based recommenders
	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/trust_based_recommenders.png)

- 关系链 Friendship Demographic methods

#### Ranking and Session Modeling
- Independent click model
- Logistic click model。Exponential family model for click; user looks at all
- Sequential click model; User traverses list
- Skip click model
- Context skip click model

### Hybrid Approaches
- Online-Nearline-Offline Recommendation（在线-近线-离线）三层混合机制
- [推荐系统中所使用的混合技术介绍](http://www.52ml.net/318.html)
- [Collaborative Filtering Ensemble]()
	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images//hybridization.png)

### 推荐算法对比

- [推荐系统的常用算法对比](http://dataunion.org/bbs/forum.php?mod=viewthread&tid=835&extra=)
- [推荐算法总结](http://blog.csdn.net/oopsoom/article/details/33740799)

常用推荐算法点评(by 陈运文)

- Item-based collaborative filtering
	- 应用最为广泛的方法
	- 存在各种计算方法的改进;但Similarity计算随意性大
- Content-based algorithm
	- 实现简单、直观,常用于处理冷启动问题
	- 推荐精度低
- Latent Factor Model
	- 单一模型效果最好的方法;但难以实时更新模型
	- KDD-Cup,Netflix Prize ...
- Statistics-based
	- 简陋,直观,非个性化,被大量使用
	- 可用于补足策略

### 效果评估
- MAP/nDCG: top-N推荐
- RMSE/MAE: 评分预测问题
- A/B Testing: 点击率、转化率

## 推荐系统实战Case

### Netflix
- top-2 algorithms：SVD，RBM
- 具体请参考[Netflix Prize and SVD](http://buzzard.ups.edu/courses/2014spring/420projects/math420-UPS-spring-2014-gower-netflix-SVD.pdf)，[The BellKor Solution to the Netflix Grand Prize](http://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf)

### twitter
[mining twitter]() #好书推荐#此书深入浅出描述如何挖掘Twitter数据， 每章有详细的文字加Python代码

### linkedIn
LinkedIn最新的推荐系统文章[The Browsemaps: Collaborative Filtering at LinkedIn](http://ceur-ws.org/Vol-1271/Paper3.pdf)。里面基本没涉及到具体算法，但作者介绍了CF在LinkedIn的很多应用，以及他们在做推荐过程中获得的一些经验。最后一条经验是应该监控log数据的质量，因为推荐的质量很依赖数据的质量！@breezedeus

### [个性化推荐技术#总结-袁全](https://breezedeus.github.io/2012/11/01/breezedeus-yuanquan-etao.html)
- 相关性推荐，点击数据更有用；补充性推荐，购买数据更有用；要根据用户行为意图选择不同的推荐方法。
- 对于不同种类的产品，当用户处在同一购物流程时，其理想的相关性推荐/补充性推荐的概率也差别很大。
- Mixture Logistic Regression

### [面向广告主的推荐，江申@百度](https://breezedeus.github.io/2012/11/10/breezedeus-jiangshen.html)
- 技术目标要正确；譬如对于拍卖词推荐，其数学目标的烟花过程为：推荐最相关的词->推荐广告主采用率最高的词->推荐采用率最高且产生推广效果最佳的词。
- 在拍卖词推荐中主要涉及到三种模型：相关性模型、采用率模型和推广效果模型。
- 负反馈：按照item已经对user展示的次数指数级降低其权重，避免同一个item多次重复被展示给一个用户。

### [个性化推荐技术#总结 稳国柱@豆瓣](https://breezedeus.github.io/2012/11/12/breezedeus-wenguozhu.html)
- 电影推荐：首先把电影按照电影标签进行分组（比如分成动作片，剧情片等）；然后在每个组里面使用CF算法产生推荐结果；最后把每组中获得的推荐按照加权组合的方式组合在一块。
- 图书推荐：图书有一定的阶梯性，在大部分的场合，我们需要的并不是与自己相似的用户的推荐，而是与自己相似的专家的推荐。
- 电台的音乐推荐：必须使用一个算法系统（其中包含多个算法）来针对不同的用户进行不同的算法调度

### [年终总结 & 算法数据的思考 by 飞林沙](http://www.douban.com/note/472267231/)

### [世纪佳缘用户推荐系统的发展历史](https://breezedeus.github.io/2015/01/31/breezedeus-review-for-year-2014-tech.html)

- "总结、温习，这两点让人成长。而不是你走得有多快！"
- 天真的算法年：item-based kNN。推荐以前看过的item的相似item。可逆（Reciprocal）推荐算法，是什么东西？[Reciprocal recommendation](http://search.aol.com/aol/search?s_it=topsearchbox.search&v_t=opensearch&q=Reciprocal+recommendation)
- 技术为产品服务，而不是直接面向用户；数据质量是地基，保证好的质量很不容易；如何制定正确的优化指标真的很难；业务理解 > 工程实现；数据 > 系统 > 算法；快速试错；
- Dirichlet Process 和 Dirichlet Process Mixture
- Alternating Direction Method of Multipliers(ADMM)
- [利用GBDT模型构造新特征](https://breezedeus.github.io/2014/11/19/breezedeus-feature-mining-gbdt.html)
- [特征哈希（Feature Hashing）](https://breezedeus.github.io/2014/11/20/breezedeus-feature-hashing.html)
- 不平衡数据的抽样方法。参考文献：William Fithian, Trevor Hastie, Local Case-Control Sampling Efficient Subsampling in Imbalanced Data Sets, 2014.
- [世纪佳缘推荐系统之我见](http://www.douban.com/note/484853135/)
	- 明确推荐评价指标：对于婚恋推荐系统来说，最核心的指标无外乎付费的转换率	- 我们倒着来推，把问题转换为识别出最愿意付费的那些用户，然后找到这些用户感兴趣的用户，通过产品引导让这些用户发信
	- 能不能从数据跳出来对产品提出一些创意性改进从而产生的产品模式和收费模式的变革。

### [New Directions in Recommender Systems](http://www.wsdm-conference.org/2015/wp-content/uploads/2015/02/WSDM-2015-PE-Leskovec.pdf)
- [飞林沙-读后总结](http://www.douban.com/note/484692347/)
- 需要理解 可替换 和 可补充，这两种推荐形式。
- 怎样生成替代品的推荐理由，应该是更好，而不是他们包含同一关键词
- 推荐一整套装备
- Inferring Networks from Opinions阅读总结：
	- Product Graph：Building networks from product text		- Understand the notions of substitute and complement goods		- Generate explanations of why certain products are preferred
		- Recommends baskets of related items
	- learn x and y are related?
		- Attempt 1: Text features；缺点：High-dimensional，Prone to overfitting，Too fine-grained
		- Attempt 2: Features from Topics。也就是把第一种方法，用topic vector替换，相当于降维了。
		- Attempt 3: Learn ‘good’ topics。Learn to discover topics that explain the graph structure；
			- Idea: Learn both simultaneously；we want to learn to project documents (reviews) into topic space such that related products are nearby；
			- Combining topic models with link prediction；topic和link利用一个目标函数，一起训练。
			- Issue 1: Relationships we want to learn are not symmetric；Solution: We solve this issue by learning “relatedness” in addition to “directedness”
			- Issue 3: The model has a too many parameters；Solution: Product hierarchy；Associate each node in the category tree with a small number of topics
			- 整个模型用EM算法来求解，类似于PLSA的EM算法。

### [美团推荐团队-机器学习中的数据清洗与特征处理综述](http://tech.meituan.com/machinelearning-data-feature-process.html)

### [美团推荐算法实践：机器学习重排序模型成亮点](http://www.csdn.net/article/2015-01-30/2823783)
- [美团推荐算法实践](http://tech.meituan.com/mt-recommend-practice.html)
- 本文介绍了美团网推荐系统的构建和优化过程中的一些做法，包括数据层、触发层、融合过滤层和排序层五个层次，采用了HBase、Hive、storm、Spark和机器学习等技术。两个优化亮点是将候选集进行融合与引入重排序模型。


### [文章-Recommending music on Spotify with deep learning](http://benanne.github.io/2014/08/05/spotify-cnns.html)
基于作者的实习经历讲Spotify的音乐推荐，内容涉及：协同过滤、基于内容的推荐、基于深度学习的品味预测、convnets规模扩展、convnets的学习内容、推荐的具体应用等

- Collaborative filtering：content-agnostic（与内容无关的），容易推荐popular items。另外，new and unpopular songs cannot be recommended，即cold-start problem。
- Content-based：tags, artist and album information, lyrics, text mined from the web (reviews, interviews, …), and the audio signal itself（e.g. the mood of the music）。
- Predicting listening preferences with deep learning。

### [推荐系统的那点事](http://www.aszxqw.com/work/2014/06/01/tuijian-xitong-de-nadianshi.html)
分析了推荐系统中使用算法的误区，确实规则带来的好处简单有效。 当一个做推荐系统的部门开始重视【数据清理，数据标柱，效果评测，数据统计，数据分析】这些所谓的脏活累活，这样的推荐系统才会有救。

### [文章 Computing Recommendations at Extreme Scale with Apache Flink and Google Compute Engine](http://data-artisans.com/computing-recommendations-with-flink.html)
Flink实例！用Flink和GAE做面向大规模数据集的协同推荐，从中可看出Flink的巨大应用潜力，文中引用的材料值得一读（作者说了，细节文章即将推出敬请期待，感兴趣请持续关注）

### [推荐系统中所使用的混合技术介绍](http://www.52ml.net/318.html)
系统架构层面一般使用多段组合混合推荐框架，算法层面则使用加权型混合推荐技术，包括LR、RBM、GBDT系列。此外还介绍分级型混合推荐技术，交叉调和技术，瀑布型混合方法，推荐基础特征混合技术，推荐模型混合技术，整体式混合推荐框架等。

### [CIKM Competition数据挖掘竞赛夺冠算法陈运文](http://www.52nlp.cn/cikm-competition-topdata)
该文讲述的不是推荐算法，而是一个分类问题，不过也有一些对我个人有启发的地方：

- 考虑到样本分布不均匀，在计算Macro Precision和Recall时，由于分母是该category的Query number，所以越是稀少的类别，其每个Query的预测精度对最终F1值的影响越大。换句话说冷门类别对结果的影响更大，需要格外关注。
- 多分类问题的处理方式，没有将跨类的样本进行拆分，而是将多类的训练样本单独作为一个类别，实践验证效果更好。
- 社交网络中的智能推荐的思想也可以在这里运用。类似推荐系统中的<User, Item>关系对，这里<Query, Title>的关系可以使用协同过滤（Collaborative Filtering）的思想，当两个Query所点击的Title列表相似时，则另外Query的category可以被“推荐”给当前Query。
- 在Ensemble框架下，分类器分为两个Level: L1层和L2层。L1层是基础分类器，前面所提到的方法均可以作为L1层分类器来使用；L2层基于L1层，将L1层的分类结果形成特征向量，再组合一些其他的特征后，形成L2层分类器（如SVM）的输入。这里需要特别留意的是用于L2层的训练的样本必须没有在训练L1层时使用过。

	在设计Ensemble L1层算法的过程中，有很多种设计思路，我们选择了不同的分类算法训练多个分类模型，而另外有队伍则为每一个类别设计了专用的二分分类器，每个分类器关注其中一个category的分类(one-vs-all)；也可以选择同一种分类算法，但是用不同的特征训练出多个L1层分类器；另外设置不同的参数也能形成多个L1层分类器等

### [学生强则国强，访天猫推荐算法大赛Top 9团队](http://www.csdn.net/article/2014-08-27/2821403-the-top-9-of-ali-bigdata-competition/9)

根据用户4个月在天猫的行为日志，预测用户u在将来一个月是否会购买某个品牌b

模型的训练思想：

由于这个问题中正负样本比例悬殊，我们使用了级联的思想过滤掉大量的样本来提升训练速度，同时也提升了模型准确率。在第一级选用训练和预测速度都较快的逻辑回归模型，过滤掉>80%的样本。在第二级选用拟合能力更强的GBRT、RF、神经网络等非线性模型。最后选用神经网络将第二级的非线性模型融合起来。

### [Large scale recommendation in e-commerce -- qiang yan](http://www.slideshare.net/scmyyan/large-scale-recommendation-in-ecommerce-qiang-yan)

Justin:online match + online learning works very well
赞processing stack。我们最近在豆瓣fm上也做了online learning，improve10个点左右

### [Google News Personalization: Scalable Online Collaborative Filtering](http://www2007.org/papers/paper570.pdf)
- 推荐的结果是三个算法的融合，即MinHash, PLSI, covisitation。融合的方式是分数线性加权。
- 一个主要的思想是“online”的进行更新，所以这个地方一定要减少规模，索引使用了User Clustering的算法，包括Min Hash和PLSI。在新数据来的时候，关键是不要去更新User Cluster，而是直接更新所属的Cluster对于URL的点击数据。对于新用户，使用covisitation的方法进行推荐
- [Personalized News Recommendation Based on Click Behavior](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.308.3087&rep=rep1&type=pdf)
- [The YouTube video recommendation system]()

## 推荐系统总结

### 实践中的关键点

#### 数据预处理
- 更多的有效数据，更好的推荐效果

#### 隐式反馈的使用
- 显式反馈(explicit feedbacks):
	- 购买、评分、接受推荐、点击喜欢。。。
	- 数量稀疏(用户是最懒的人)
- 隐式反馈(implicit Feedbacks):
	- 浏览、收听、点击、下载。。。
	- User/item相关的profile、keyword、tags
	- 反馈中占大多数(往往被忽略)
- 如何利用好隐式反馈?
	- 对提高推荐精度有良好效果(SVD->SVD++)

#### SNS关系的使用
- SNS关系包括：
	- follower/followee，好友，群关系
	- user-user actions，e.g. "retweet/at/comment..."
- SNS关系的使用
	- 用于user之间的关系计算(graph-based)
	- 作为隐式反馈，用于SVD++

#### 时间因素的使用
- user的行为受时间影响
- Item的状态也受时间影响

#### 利用地域信息
- 特定的应用场景
- 基于规则
- 基于地域信息的关联规则挖掘
- Item-based协同过滤，item similarity计算时加入距离属性
- Latent factor，user-location作为隐式反馈使用

#### User冷启动的处理
- 热门推荐(排行榜)永远都是一个可用的方案
	- 点击总量最多
	- 最近点击最多
	- 评分最高
- 充分利用任何用户信息
	- 性别、年龄
	- 来自其他应用的数据
- 口味测试
	- 有代表性的选项
	- 热门/大部分用户熟知的选项
	- 有区分度的选项

#### Item冷启动的处理
- Content-basedmethods永远都是一个可用方案
	- Category
	- Tags
	- Topic
- 相关技术(NLP、ML)
	- 自动分类
	- 自动标签提取
- 倒排索引的使用
	- 适用于item数量庞大
	- 索引的查询与合并


### 总结
- 技术目标要正确。

	举个例子，搜索广告中的关键词推荐，技术目标是什么？是相关性足够好，还是采用率够高，或者是广告主采用后收益高。这里首先需要确定好，如果优化目标错误了，那接下来的工作都要打折扣。假设关键词推荐的技术目标是：推荐采用率最高且产生推广效果最佳的词。那么在做推荐时，可以用一个模型解决上述目标，也可以用多个模型。一般情况下，将优化目标拆分成多个模型，将减小复杂度。譬如在关键词推荐里，就有三个模型：相关性模型、采用率模型和推广效果模型。

	1. 首先要定义好什么叫”好的推荐“，这是解决任何一个技术问题的前提。
	2. 在有了明确的定义之后，实际问题一般会蜕变为一个优化问题，用数学工具给出它最好的解答。
	3. 数学上的解答可能在技术上无法实现，或者说有可能复杂度太高，那么需要一个比较好的近似解。不要小看这一步，大部分问题出在这里。
	4. 迭代改进。算法实现以后可能实际表现与预想的不同，需要重新定义”好的推荐“。这样一个周期下来，推荐效果应当有肉眼可见的改进。

- 推荐算法选择，要根据数据来做选择。
	- 各取所长，互相补充
	- 算法无好坏之分，只有是否合适

- 再总结几点：
	- 推荐不应该是推荐算法，而应该是推荐产品。本文的标题虽为推荐算法总结，但这是无米之炊。
	- 产品是1，算法是0。没有合适的产品之前，算法对用户几乎不产生什么价值，一旦产品成立，算法能让产品实现质的飞跃。
	- 有些业务提升不牵涉到任何的模型改进，但需要算法人员保持对业务的关注和理解，而不是一直躲在后面。
	- 推荐技术是否能成就一个伟大的产品，不会。基于推荐技术对产品的强烈依附关系，它不会反过来促成一个产品，但它终将成就一种用户习惯。
	- 不要空谈算法，要根据不同的产品场景。图书更强调个性化，电影更强调热门与根据标签筛选。
	- 算法效果的度量方式往往决定了你努力的方向，面向不同类型的推荐，其中一个重要措施，就是不要采用同样一套标准去衡量你的工作结果。在实践中永远只看到ctr、precision/recall、rmse那样的衡量指标。那是衡量一个单一算法，而不是一个推荐系统的指标，或者说，不是衡量跟它绑定在一起的推荐产品的指标。
	- 个性化，是要让所有人都满意，而不是为了统计上80%人的体验而牺牲掉剩下20%人的体验。
	- 算法人员更应该站出来，主动填补算法与产品之间的空隙。**要做一个能用产品语言说话的算法工程师**。
	- 目前大部分的优化都集中在算法层面上，而忽略了数据优化。数据优化的增益能极大提升算法优化的上界。

- [Recommendation Systems: What developments have occurred in recommender systems after the Netflix Prize?](http://www.quora.com/Recommendation-Systems/What-developments-have-occurred-in-recommender-systems-after-the-Netflix-Prize/answer/Xavier-Amatriain?srid=z0Q5&share=1)

	下面是Xavier Amatriain在Quora上关于目前推荐系统研究总结，涵盖了推荐系统的多样性，基于上下文环境推荐，社交信息的引入。其中他谈到，评分预测已经不是主流，LTR的应用会更符合推荐的初衷。

	- Implicit feedback from usage has proven to be a better and more reliable way to capture user preference.
	- Rating prediction is not the best formalization of the "recommender problem". Other approaches, and in particular personalized Learning to Rank, are much more aligned with the idea of recommending the best item for a user.
	- It is important to find ways to balance the trade-off between exploration and exploitation. Approaches such as Multi-Armed Bandit Algorithms offer an informed way to address this issue.
	- Issues such as diversity, and novelty can be as important as relevance.
	- It is important to address the presentation bias caused by users only being able to give feedback to those items previously decided where good for them.
	- The recommendation problem is not only a two dimensional problem of users and items but rather a multi-dimensional problem that includes many contextual dimensions such as time of the day or day of the week. Algorithms such as Tensor Factorization or Factorization Machines come in very handy for this.
	- Users decide to select items not only based on how good they think they are, but also based on the possible impact on their social network. Therefore, social connections can be a good source of data to add to the recommendation system.
	- It is not good enough to design algorithms that select the best items for users, these items need to be presented with the right form of explanations for users to be attracted to them.

## 其他资料
- 一个基于Python的Nearest Neighbors Search库 http://t.cn/RvVr8Gb。博文介绍 http://t.cn/RwSZk2l。另外基于这个库的一个推荐系统 http://t.cn/RZ339rQ，将作为一个demo presentation在WWW上出现。作者是剑桥的博士后 @唧唧歪歪de计算机博士

- 国外在线交友网站eHarmony是如何做用户推荐的：http://t.cn/RwLG7e9，对应视频下载：http://t.cn/RwQR5xJ。里面干活不少，比如介绍了如何从照片中抽取特征。很多地方跟我们的思路其实挺像的：http://t.cn/RwV1BQh。另外，V r Hiring More ...（推荐、机器学习、分布式系统、php）
- [文章]《Personalized Recommendations at Etsy》http://t.cn/Rz7MdpO 介绍Etsy采用的个性化推荐算法，包括矩阵分解、交替最小二乘、随机SVD和局部敏感哈希等


## 参考文献
1. [数据挖掘技术在推荐系统的应用，陈运文](http://wenku.baidu.com/view/0607e780d0d233d4b14e699e.html)
2. [Google News Personalization: Scalable Online Collaborative Filtering](http://www2007.org/papers/paper570.pdf)
3. [Tutorial: Recommender Systems; Dietmar Jannach](http://ijcai13.org/files/tutorial_slides/td3.pdf)
4. [Application of Dimensionality Reduction in Recommender System -- A Case Study](http://ai.stanford.edu/~ronnyk/WEBKDD2000/papers/sarwar.pdf)
5. [Up Next: Retrieval Methods for Large Scale Related Video Suggestion](http://vdisk.weibo.com/s/DaKXoKQC5TSH)
6. [Alex-recommendation](http://alex.smola.org/teaching/berkeley2012/slides/8_Recommender.pdf)
7. [Large scale recommendation in e-commerce -- qiang yan](http://www.slideshare.net/scmyyan/large-scale-recommendation-in-ecommerce-qiang-yan)
8. [Recommender System slices. MLSS14](http://www.slideshare.net/xamat/recommender-systems-machine-learning-summer-school-2014-cmu)
9. [Recommender System video. MLSS14](http://videolectures.net/kdd2014_amatriain_mobasher_recommender_problem/)
10. [Context Aware Recommendation. Bamshad Mobasher](http://www.kdd.org/kdd2014/tutorials/KDD-%20The%20RecommenderProblemRevisited-Part2.pdf)
11. [KDD - The Recommender Problem Revisited](http://www.kdd.org/kdd2014/tutorials/KDD%20-%20The%20Recommender%20Problem%20Revisited.pdf)
12. [推荐系统的资料分享](http://blog.sina.com.cn/s/blog_804abfa70101btrv.html)，里面分享的几篇文章值得一看
13. [寻路推荐-理念篇](http://www.wentrue.net/blog/?p=1621)
14. [精准定向的广告系统 yiwang](http://www.docin.com/p-936085086.html)
15. [推荐系统读物](http://bigdata.memect.com/?p=11684)
16. [Moving Beyond CTR: Better Recommendations Through Human Evaluation](http://blog.echen.me/2014/10/07/moving-beyond-ctr-better-recommendations-through-human-evaluation/)
