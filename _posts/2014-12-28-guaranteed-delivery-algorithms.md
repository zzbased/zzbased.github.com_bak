---
layout: post
title: "guaranteed delivery algorithms"
description: ""
category:
tags: []
---


首先合约广告分配问题的输入，可以用一个二部图(bipartite graph)来表示，如下图所示。图的左边是supply，也就是impressions。这里我们根据定向人群将impression划分成不同的簇，每个簇上的数值表示该定向人群下的impression总数；图的右边是demand，即合约广告，其数值表示该合约广告所需要曝光的量。

![](contract_ad_bipartite.jpg)

