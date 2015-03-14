---
layout: post
title: "guaranteed delivery algorithms"
description: ""
category:
tags: []
---

# 合约cpm广告算法解析

## 什么是合约cpm广告

## 合约cpm广告数学模型

合约cpm广告分配问题的输入，可以用一个二部图(bipartite graph)来表示，如下图所示。
图的左边是supply nodes，也就是impressions。我们根据定向人群将impression划分成不同的簇，每个簇上的数值表示该定向人群下的impression预估总数，以s_i表示；图的右边是demand nodes，即合约广告，其数值表示该合约广告所需要曝光的量，以d_j表示。
如果supply i 满足合约广告 j 的定向需求，则在s_i 与d_j 之间添加一条直线(edge)，表示是可match的。给每条直线一个权重x_{i,j}，表示supply i分配给合约广告 j的比例。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_post/images/contract_ad_bipartite.jpg)

我们的优化目标是：找到一个合理的分配方案，使得满足下面的条件：

![](images/gd_allocation_target.png)

上式中：\tau(i)表示所有与supply i有连接的合约广告集合，\tau(j)表示所有满足合约广告j的supply集合。

合约cpm播放的目标有两个：(1)最大化展现效率；(2)最小化补偿。最大化展现效率就是尽可能让所有demand端需要的曝光都能得到满足；最小化补偿和最大化展现效率意思一致，就是让未能足量曝光导致的补偿尽可能少。
把上述目标进一步优化后，得到进阶的目标函数：

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_post/images/gd_allocation_target2.png)

上面式子中，第(1)个约束条件是demand constraints，第(2)个约束条件是supply constraints，第(3)个约束条件是非负性约束。

## 合约cpm广告系统模型

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_post/images/gd_system_architecture.png)

## HWM算法

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_post/images/hwm_algorithms.png)

## Shale算法

