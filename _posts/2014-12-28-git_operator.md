---
layout: post
title: "git相关操作指南"
description: ""
category:
tags: [git]
---

## git相关操作指南

###最基础的git操作
- 创建新仓库。

> 创建新文件夹，打开，然后执行 git init 以创建新的 git 仓库。

- 检出仓库。

> 执行如下命令以创建一个本地仓库的克隆版本：
git clone /path/to/repository
如果是远端服务器上的仓库，你的命令会是这个样子：
git clone username@host:/path/to/repository

- 工作流。

> 你的本地仓库由 git 维护的三棵“树”组成。第一个是你的 工作目录，它持有实际文件；第二个是 缓存区（Index），它像个缓存区域，临时保存你的改动；最后是 HEAD，指向你最近一次提交后的结果。

- 添加与提交。

> 你可以计划改动（把它们添加到缓存区），使用如下命令：
git add <filename>
git add *
这是 git 基本工作流程的第一步；使用如下命令以实际提交改动：
git commit -m "代码提交信息"
现在，你的改动已经提交到了 HEAD，但是还没到你的远端仓库。

- 推送改动。

> 你的改动现在已经在本地仓库的 HEAD 中了。执行如下命令以将这些改动提交到远端仓库：
git push origin master
可以把 master 换成你想要推送的任何分支。
如果你还没有克隆现有仓库，并欲将你的仓库连接到某个远程服务器，你可以使用如下命令添加：
git remote add origin <server>
如此你就能够将你的改动推送到所添加的服务器上去了。

- 更新与合并。

> 要更新你的本地仓库至最新改动，执行：
git pull
以在你的工作目录中 获取（fetch） 并 合并（merge） 远端的改动。
要合并其他分支到你的当前分支（例如 master），执行：
git merge <branch>
两种情况下，git 都会尝试去自动合并改动。不幸的是，自动合并并非次次都能成功，并可能导致 冲突（conflicts）。 这时候就需要你修改这些文件来人肉合并这些 冲突（conflicts） 了。改完之后，你需要执行如下命令以将它们标记为合并成功：
git add <filename>
在合并改动之前，也可以使用如下命令查看：
git diff <source_branch> <target_branch>

- 替换本地改动。

> 假如你做错事（自然，这是不可能的），你可以使用如下命令替换掉本地改动：
git checkout -- <filename>
此命令会使用 HEAD 中的最新内容替换掉你的工作目录中的文件。已添加到缓存区的改动，以及新文件，都不受影响。
假如你想要丢弃你所有的本地改动与提交，可以到服务器上获取最新的版本并将你本地主分支指向到它：
git fetch origin
git reset --hard origin/master

- git与svn命令的区别。更多请参考[Subversion 用户眼中的 Git](http://www.uml.org.cn/pzgl/201211265.asp)

> git pull -- svn up；
> git add -- svn add；
> git commit -- svn ci；
> git clone -- svn cp；
> git checkout -- svn co；
> git push -- 无；
> git status -- svn status；
> git revert -- svn revert；
> git diff --  svn diff；
> git merge -- svn merge；


### git 客户端
常见的git客户端有：[msysgit](https://msysgit.github.io/)，[TortoiseGit](https://code.google.com/p/tortoisegit/)

我常用的是msysgit。它自带一个git gui。下图中，缓存改动的命令为git add，提交的命令为git commit，上传的命令为git push。

![git_gui](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/git_gui.png)

在公司里用git，可能需要设置git proxy，命令如下面所示：
git config --global http.proxy https://web-proxyhk.oa.com:8080

git gui默认不保存用户名和密码，如果需要让它保存，可以设置credential.helper，
git config --global credential.helper "cache --timeout=3600"。具体请参考[skip-password](http://stackoverflow.com/questions/5343068/is-there-a-way-to-skip-password-typing-when-using-https-github)


### git submodule

[Git 工具 - 子模块](http://git-scm.com/book/zh/v1/Git-工具-子模块)

经常有这样的事情，当你在一个项目上工作时，你需要在其中使用另外一个项目。也许它是一个第三方开发的库或者是你独立开发和并在多个父项目中使用的。

Git 通过子模块处理这个问题。子模块允许你将一个 Git 仓库当作另外一个Git仓库的子目录。这允许你克隆另外一个仓库到你的项目中并且保持你的提交相对独立。

git submodule 就是 svn:externals 翻版

**克隆一个带子模块的项目**，当你接收到这样一个项目，你将得到了包含子项目的目录，但里面没有文件。
此时应该先调用 git submodule update --init

git submodule其他参考：

- [Git Submodule使用完整教程](http://www.kafeitu.me/git/2012/03/27/git-submodule.html)
- [Git Submodule的坑](http://blog.devtang.com/blog/2013/05/08/git-submodule-issues/)

### 如何查看repo url

类似于svn info的功能。

If referential integrity has been broken:
git config --get remote.origin.url

If referential integrity is intact:
git remote show origin

参考自[How can I determine the url that a local git repo was originally cloned from?](http://stackoverflow.com/questions/4089430/how-can-i-determine-the-url-that-a-local-git-repo-was-originally-cloned-from)

### 其他参考资料

- [完整的git教程](http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)
- [git使用简易指南](http://www.bootcss.com/p/git-guide/)
- [git参考手册](http://gitref.org/zh/basic/#status)
- [Git冲突：commit your changes or stash them before you can merge. 解决办法](http://www.letuknowit.com/post/144.html)
- [git与github的关联](http://blog.csdn.net/authorzhh/article/details/7533086)

