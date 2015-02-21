---
layout: post
title: "开发我的第一个ios程序"
description: ""
category:
tags: [ios, objective-c]
---

## 开发我的第一个ios程序 ##


参考[这篇文章 a weather app case study part1](http://www.raywenderlich.com/55384/ios-7-best-practices-part-1)，[a weather app case study part2](http://www.raywenderlich.com/55386/ios-7-best-practices-part-2)，开发我的第一个iOS程序。

### cocoapods ###
根据这篇文章的指南，我们首先需要安装[cocoapods](http://cocoapods.org)。cocoapods是一个针对objective-c工程的依赖管理器(dependency manager for Objective-C projects)，类似于pig/eggs之于python，gems之于ruby，maven之于java，cpan之于perl。

题外话，为什么c++没有类似cocoapods的库，这里有一篇[讨论](http://programmers.stackexchange.com/questions/170679/why-are-there-no-package-management-systems-for-c-and-c)，可以看看。

回到cocoapods的安装，直接执行下面这条命令即可，sudo gem install cocoapods (之前文章里，刚使用gem安装了jekyll)。安装完成后，执行which pod，会出现"/usr/bin/pod"的字样。

再接着，利用xcode5创建一个空的ios工程:SimpleWeather。在teminal里，cd到SimpleWeather目录，利用pico编辑器编写podfiles。

platform :ios, '7.0'
pod 'Mantle'
pod 'LBBlurredImage'
pod 'TSMessages'
pod 'ReactiveCocoa'

目的是利用pod来安装 Mantle, LBBlurredImage, TSMessages, ReactiveCocoa这几个库。执行pod install即可安装。安装完成后如下所示：
![pod install over](http://d.pr/i/aTyV+)


### main view controller ###
打开SimpleWeather.xcworkspace。新建一个UIViewController的子类。相当于是mvc模型中的控制器。
[UIViewController](https://developer.apple.com/library/ios/Documentation/UIKit/Reference/UIViewController_Class/Reference/Reference.html)：A view controller manages a set of views that make up a portion of your app’s user interface.
As part of the controller layer of your app, a view controller coordinates its efforts with model objects and other controller objects — including other view controllers — so your app presents a single coherent user interface.

Where necessary, a view controller:
1.resizes and lays out its views
2.adjusts the contents of the views
3.acts on behalf of the views when the user interacts with them

![mvc模型](http://d.pr/i/qEs+)

接下去就按照文章的演示继续往下做。Setting Up Your App’s Views。


### retrieve data & model ###
to be continue;




###reference:###
[免费的ios7教程](https://leanpub.com/ios7daybyday)

[ios wiki](http://www.ios-wiki.com)

[iOS设计新手指导](http://www.cocoachina.com/newbie/basic/2013/1225/7607.html)

[马上着手开发 iOS 应用程序](https://developer.apple.com/library/ios/referencelibrary/GettingStarted/RoadMapiOSCh/chapters/RM_YourFirstApp_iOS/Articles/01_CreatingProject.html)

[分布式系统的事务处理](http://coolshell.cn/articles/10910.html)

