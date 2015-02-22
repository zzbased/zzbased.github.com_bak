---
layout: post
title: "利用jekyll+github搭建简易博客"
description: ""
category:
tags: [blog, git]
---

##利用jekyll+github搭建简易博客##

### 包管理系统（package management system）
包管理系统就是指某个平台上对第三方包进行统一组织管理的系统。一个包管理系统一般由以下几部分组成：

- 一套包的制作、描述、配置、发布 规范
- 用于包的在线发布、管理的 仓库（repository），也就是包的 源（source）/注册处（registry）/索引（index）/展馆（gallery）
- 包的本地安装、配置、升级、卸载 工具

[List of software package management systems](http://en.wikipedia.org/wiki/List_of_software_package_management_systems)

- mac os：
	- MacPorts - Formerly known as DarwinPorts：port。
- python：
	- pip - a package manager for Python and the PyPI programming library
	- EasyInstall - a package manager for Python and the PyPI programming library which is part of the Setuptools packaging system
	- anaconda：Anaconda is a distribution of the Python programming language that aims to simplify package management and deployment. Its package management system is conda

- ruby：
	- gem：RubyGems，用于ruby程序和程序库的一套打包系统，它让开发人员可以把自己的ruby程序库打包成一种易于维护和安装的形式。

- perl：
	- CPAN：a programming library and package manager for Perl

- Node.js
	- npm: a programming library and package manager for Node.js

- Java
	- Maven: a package manager and build tool for Java

### Jekyll步骤

[Jekyll theme](http://jekyllthemes.org)

[Using Jekyll with Pages](https://help.github.com/articles/using-jekyll-with-pages/)

[在Github上搭建Jekyll博客和创建主题](http://yansu.org/2014/02/12/how-to-deploy-a-blog-on-github-by-jekyll.html)

[Jekyll中使用MathJax](http://www.pkuwwt.tk/linux/2013-12-03-jekyll-using-mathjax/)

- 在github上创建username.github.com目录。
- 安装jekyll，然后jekyll new username.github.com，将这些内容git push到github。

### Jekyll-bootstrap步骤 ###
1. Github工作目录：
搭建Github工作目录，需要先把ssh通道建立好，参看下面两篇文章。[产生ssh keys](https://help.github.com/articles/generating-ssh-keys), [可能碰到的问题](https://help.github.com/articles/error-permission-denied-publickey)

2. markdown编辑器：
在macbook上，我使用的编辑器是lightpaper. 引用图像存储链接服务是 [droplr](droplr.com)

3. 我使用的是[jekyllbootstrap](http://jekyllbootstrap.com)。号称三分钟可以教会搭建github博客，事实就是如此。参考这篇入门指南即可。[入门指南](http://jekyllbootstrap.com/usage/jekyll-quick-start.html)

4. 需要注意的是，如果在上面准备工作里github的ssh设置没能成功。
	git remote set-url origin git@github.com:zzbased/zzbased.github.com.git
	可以更改为https地址:
	git remote set-url origin https://github.com/zzbased/zzbased.github.com.git

5. 安装好jekyll后，就可以本地调试。我们利用index.md，可以在原基础上做修改即可。

6. 然后在_post文件夹里，删除原来的example。利用rake post title="xxx"新增一个md文件。接下来就开始编辑了。

7. 如果不喜欢页面最下面的footer, 可以在“./_includes/themes/twitter/default.html”文件中，把footer屏蔽掉。不过建议还是留着，可以让更多的人接触到这项工具。

8. 在本地执行 jekyll serve,然后就可以在本机浏览器上通过0.0.0.0:4000预览网站。
