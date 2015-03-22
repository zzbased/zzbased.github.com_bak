---
layout: post
title: "gbdt_adaboost_bootstrap"
description: ""
category:
tags: []
---


# gbdt，adaboost，bootstrap

## 决策树

Split training set at ”the best value” of ”the best feature”

- Information gain (ratio)
- Gini index
- Mean square error

## boosting方法共性

- Train one base learner at a time.
- Focus it on the mistakes of its predecessors.
- Weight it based on how ‘useful’ it is in the ensemble (not on its training error).

## gbdt

意为 gradient boost decision tree。又叫MART（Multiple Additive Regression Tree)

**好好看一下 kimmy的ppt: Gradient Boosted Decision Tree**

### 分类树和回归树

- 分类树：预测分类标签；C4.5；选择划分成两个分支后熵最大的feature；
- 回归树：预测实数值；回归树的结果是可以累加的；最小化均方差；

### Shrinkage

Shrinkage（缩减）的思想认为，每次走一小步逐渐逼近结果的效果，要比每次迈一大步很快逼近结果的方式更容易避免过拟合。即它不完全信任每一个棵残差树，它认为每棵树只学到了真理的一小部分，累加的时候只累加一小部分，通过多学几棵树弥补不足。

## adaboost

Adaptive Boosting；

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_post/images/adaboost_pseudo_code.png)

**Loss Function**

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_post/images/adaboost_lost_function.png)

为什么选择Exponential Loss？

- Loss is higher when a prediction is wrong.
- Loss is steeper when a prediction is wrong.
- Precise reasons later

[Introduction to Boosting](http://www.cs.man.ac.uk/~stapenr5/boosting.pdf) 稍后再仔细读下

## bootstrap

## boosting方法比较

- gbdt的核心在于：每一棵树学的是之前所有树的结论和残差。每一步的残差计算其实变相地增大了分错instance的权重，而已经分对的instance则都趋向于0。
- Adaboost：是另一种boost方法，它按分类对错，分配不同的weight，计算cost function时使用这些weight，从而让“错分的样本权重越来越大，使它们更被重视”。
- Bootstrap也有类似思想，它在每一步迭代时不改变模型本身，也不计算残差，而是从N个instance训练集中按一定概率重新抽取N个instance出来（单个instance可以被重复sample），对着这N个新的instance再训练一轮。由于数据集变了迭代模型训练结果也不一样，而一个instance被前面分错的越厉害，它的概率就被设的越高，这样就能同样达到逐步关注被分错的instance，逐步完善的效果。

## 总结

[集成学习：机器学习刀光剑影 之 屠龙刀](http://www.52cs.org/?p=383)

- Bagging和boosting也是当今两大杀器RF（Random Forests）和GBDT（Gradient Boosting Decision Tree）之所以成功的主要秘诀。
- Bagging主要减小了variance，而Boosting主要减小了bias，而这种差异直接推动结合Bagging和Boosting的MultiBoosting的诞生。参考:Geoffrey I. Webb (2000). MultiBoosting: A Technique for Combining Boosting and Wagging. Machine Learning. Vol.40(No.2)
- LMT(Logistic Model Tree ) 应运而生，它把LR和DT嫁接在一起，实现了两者的优势互补。对比GBDT和DT会发现GBDT较DT有两点好处：1）GBDT本身是集成学习的一种算法，效果可能较DT好；2）GBDT中的DT一般是Regression Tree，所以预测出来的绝对值本身就有比较意义，而LR能很好利用这个值。这是个非常大的优势，尤其是用到广告竞价排序的场景上。
- 关于Facebook的GBDT+LR方法，它出发点简单直接，效果也好。但这个朴素的做法之后，有很多可以从多个角度来分析的亮点：可以是简单的stacking，也可以认为LR实际上对GBDT的所有树做了选择集成，还可以GBDT学习了基，甚至可以认为最后的LR实际对树做了稀疏求解，做了平滑。

## 更多学习资料
- [Gbdt迭代决策树入门教程](http://suanfazu.com/t/gbdt-die-dai-jue-ce-shu-ru-men-jiao-cheng/135)
- [Boosting Decision Tree入门教程](http://www.schonlau.net/publication/05stata_boosting.pdf)
- [LambdaMART用于搜索排序入门教程](http://research.microsoft.com/pubs/132652/MSR-TR-2010-82.pdf)
- [文章]《Ask a Data Scientist: Ensemble Methods》http://t.cn/RwoVO5O “Ask a Data Scientist.”系列文章之Ensemble Methods，通俗程度可以和昨天介绍的Quora随机森林解释相媲美，但更为详尽，对常用Ensemble框架及其特点也进行了介绍，很好
