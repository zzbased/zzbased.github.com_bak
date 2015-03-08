---
layout: post
title: "语义分析方法参考文献"
description: ""
category:
tags: [machine learning]
---

# 语义分析方法参考文献


### 深度学习方法的两个应用

What we’ll do: Summarize 2 ways Deep/Neural ideas can be used

- As non-linear classifier
- As distributed representation

- Exploiting Non-linear Classifiers
It’s possible to directly apply Deep Learning to text problems with little modification, as evidenced by [Glorot et al., 2011]
But sometimes NLP-specific modifications are needed, e.g. training objective mismatch in Machine Translation N-best experiment

- Exploiting Distributed Representation
Distributed Representation is a simple way to improve robustness of NLP, but it’s not the only way (POS tagging experiment)
Promising direction: distributed representations beyond words, considering e.g. compositionality [Socher et al., 2013a]

### slides: deep learning for natural language processing and machine translation

- Language Models (LM) using Neural Nets。见72页。
- Recurrent Neural Net Language Models [Mikolov et al., 2010]。见75页。
- deep learning for natural language processing参考文献。P99。
	- Sequence labeling
		POS tagging & Name Entity Recognition [Turian et al., 2010, Collobert et al., 2011, Wang and Manning, 2013, ]
		Qi et al., 2014: A deep learning framework for character-based information extraction. Ma et al., 2014: Tagging the web: Building a robust web tagger with neural network.
		Tsuboi, 2014: Neural networks leverage corpus-wide information for part-of-speech tagging.
		Guo et al., 2014: Revisiting embedding features for simple semi-supervised learning.

	- Word Segmentation [Zheng et al., 2013, Pei et al., 2014]

		Max-margin tensor neural network for chinese word segmentation.
		Deep learning for Chinese word segmentation and POS tagging.

	- Semantics

		Word Representations [Tsubaki et al., 2013, Srivastava et al., 2013, Rockt ̈aschel et al., 2014, Baroni et al., 2014, Hashimoto et al., 2014, Pennington et al., 2014, Neelakantan et al., 2014, Chen et al., 2014b, Milajevs et al., 2014]
		Evaluating neural word representations in tensor-based compositional settings.
		A unified model for word sense representation and disambiguation.
		Efficient non-parametric estimation of multiple embeddings per word in vector space.
		Glove: Global vectors for word representation.
		Jointly learning word representations and composition functions using predicate-argument structures.
		Don’t count, predict! a systematic comparison of context-counting vs. context-predicting semantic vectors.


### 自然语言处理的基本任务
自然语言（Natural Language）其实就是人类语言，自然语言处理（NLP）就是对人类语言的处理，当然主要是利用计算机。自然语言处理是关于计算机科学和语言学的交叉学科，常见的研究任务包括：

- 分词（Word Segmentation或Word Breaker，WB）

- 信息抽取（Information Extraction，IE）：命名实体识别和关系抽取（Named Entity Recognition & Relation Extraction，NER）

- 词性标注（Part Of Speech Tagging，POS）

- 指代消解（Coreference Resolution）

- 句法分析（Parsing）

- 词义消歧（Word Sense Disambiguation，WSD）

- 语音识别（Speech Recognition）

- 语音合成（Text To Speech，TTS）

- 机器翻译（Machine Translation，MT）

- 自动文摘（Automatic Summarization）

- 问答系统（Question Answering）

- 自然语言理解（Natural Language Understanding）

- OCR

- 信息检索（Information Retrieval，IR）

### 传统文本语义分析

【Query意图分析：记一次完整的机器学习过程（scikit learn library学习笔记）】http://t.cn/RvUNAsG 博客园zero_learner的博文。利用Python机器学习包Scikit Learn具体解决Query意图问题，是一个机器学习实践的很好示例。另外作者推荐阅读相关文章“如何选择机器学习分类器”：http://t.cn/RvA6amn

@刘知远THU:略短啊，相对来讲个人感觉北大赵鑫 @BatmanFly 对LDA的总结更全面系统一些：http://t.cn/aBmeRA
http://net.pku.edu.cn/~zhaoxin/Topic-model-xin-zhao-wayne.pdf

@Wenpeng_Yin 推荐一个NLP工具: http://t.cn/8s0E6w6 中文,英文,西班牙文, 法语,德语的lemmatizer, POS tagging, denpendency parsing以及Morphologic Tagging一条龙服务, 简单易用,效果不输stanford

转//@徐君_: SMIR 2014的邀请报告云集了IR和NLP的大牛，UMass的Bruce Croft教授、微软研究院的Jianfeng Gao研究员、CMU的Ed Hovy教授、马里兰大学的Doug Oard教授和阿姆斯特丹大学的Maarten de Rijek教授将与大家探讨信息检索和自然语言处理中的语义匹配，敬请期待！
@徐君_ 语义匹配(Semantic Matching)是信息检索与自然语言处理的核心问题之一，欢迎大家关注与投稿SIGIR 2014 Workshop on Semantic Matching in Information Retrieval (SMIR 2014) http://t.cn/8sVnfFi 。时间：7月11日，地点：澳大利亚黄金海岸，投稿截止日期：5月10日。 @李航博士

### 基于深度学习的文本语义分析
本小节主要内容：rnnlm，word embeddings，word2vec，sentence-vector，glove。

word embeddings：Distributed representation can be used as pre-training of deep learning to build features of machine learning tasks as a unified model to integrate heterogeneous information (text, image, ...)

- ML = Representation + Objective + Optimization
- Good Representation is Essential for Good Machine Learning
- Semantic Hierarchy Extraction
- Cross-lingual Joint Representation
- Visual-Text Joint Representation

- Replicated Softmax: an Undirected Topic Model (NIPS 2010)
- A Deep Architecture for Matching Short Texts (NIPS 2013)
- Modeling Documents with a Deep Boltzmann Machine (UAI 2013)
- A Convolutional Neural Network for Modelling Sentences(ACL 2014)

放弃词向量(WordEmbedding)工作，直接在词袋模型(BoW)上用卷积神经网络做文本分类任务(CNN for Text Categorization)，看这篇文章的工作（待评论。。），http://t.cn/RZbN2b6 提出seq-CNN与bow-CNN两种模型直接在BoW上使用卷积层，同state-of-the-art方法相比取得较好结果

继续来广告一把和@BB-Rev 的一个用deep learning做文本推荐的工作：http://t.cn/RhJKetG 我们在这个工作里把dl和矩阵分解结合，目标不再是之前dl的分类或回归。基本思想是利用dl去学习文本(item)上的feature，mf去学习user之间的偏好，两者通过gradient交互。结果比KDD11 Blei的best paper好了一个量级

开源Recurrent Neural Network库RNNSharp. 支持标准RNN和RNN-CRF，可用于分类，序列标注等任务。支持模板特征，上下文特征，word embedding以及运行时特征。地址：http://t.cn/RvFJHPj 求推广 @52nlp

lda结构是word-hidden topic。类lda结构假设在topic下产生每个word是条件独立而且参数相同。这种假设导致参数更匹配长文而非短文。这篇文章提出word-hidden topic-hidden word，其实是(word,hidden word)-hidden topic。增加的hidden word平衡了参数对短文的适配，在分类文章数量的度量上更好很自然。
@王威廉 经过不懈的努力，Geoff Hinton及其弟子终于用Deep Boltzmann Machine捣鼓出了类似LDA的隐变量文本模型，号称其抽取的特征在文本检索与文本分类上的结果比LDA好。UAI2013论文：http://t.cn/zQbzwVi

@张俊林say 最近几个月比较关注深度学习在自然语言处理方面的应用，虽然总体而言DL在NLP并未像图像和语音识别领域一样展现出突破性的进展，但是我个人比较看好这个方向的发展前景，把目前主流的相关工作梳理了一下，后续会不断维护版本更新以及我们在这方面的探索结果。http://t.cn/Rz4Jkia

@王威廉 对命名实体识别与消歧(Named Entity Recognition/Disambiguation)与词义消歧(Word Sense Disambiguation)有兴趣的同学注意了：ACL主席Gertjan van Noord等欧洲NLP科学家近日编撰了一篇非常棒的报告，汇总了大量的实体识别与消歧，词义消歧数据集，工具集。强烈推荐！http://t.cn/RvzajR2

Yoav Goldberg写了个测评文档，大致结论就是GloVe和word2vec如果正常比的话 效果差不多，没有宣称的11%这么大。。 链接：http://t.cn/RP0gMXB
@刘知远THU
斯坦福Richard Socher在EMNLP2014发表新作：GloVe: Global Vectors for Word Representation 粗看是融合LSA等算法的想法，利用global word co-occurrence信息提升word vector学习效果，很有意思，在word analogy task上准确率比word2vec提升了11%。 http://t.cn/RPohHyc

### 基于深度学习的图片语义分析
本节主要内容：cnn分类，cnn-sentences。

ImageNet的创始人、大规模图像分类竞赛ILSVRC的组织者、斯坦福CV组的Li Fei-Fei等人写了一篇比较详尽的ImageNet综述文章，总结了2010-2014年的成绩: http://t.cn/RhyGbGi

@沈复民 CVPR'13 best paper由google的论文fast detection of 100,000 categories on a single machine取得! 这个结果大家都猜到了！ 我在:http://t.cn/zHDS9R6

@陈博兴-NLP 最新的基于深度学习的机器翻译技术用RNN将源语言句子转换成一个vector representation，然后用另外一个RNN生成目标语言句子。Google借用这个idea，把第一个RNN换成Deep CNN，用来将图像翻译成文字，取得了很好的BLEU score。看看图片中例子，简直太赞了！http://t.cn/Rz7INhu

@王威廉 Google大神Jeff Dean月初在#CIKM2014#上关于谷歌深度学习的幻灯片，包括语音、NLP、视觉、MT以及多模态学习的最新进展: http://t.cn/RzPjoaD

@Jay_GraphLab 深度学习很强大，但训练起来非常昂贵，费时费力。然而并非每个问题都必须从头开始训练一个深度模型。这篇博客 http://t.cn/RzFAahl 详细介绍了为何以及如何使用深度学习做特征提取，并应用到简单机器视觉问题。在imagenet上训练好的模型，通过特征提取，应用到CIFAR10，可轻松达到94％以上测试准确率。

@丕子 Translating Videos to Natural Language http://t.cn/RzrJYnL 扫了一眼，做的还是image的，几乎没有video的信息嘛

@星空下的巫师 VGG已经放出他们ImageNet 2014模型的细节了， Very Deep Convolutional Networks for Large-Scale Image Recognition， see：http://t.cn/RhbZqQr

Jeff Hinton组把deep CNN(CovNets)在ImageNet上train好的模型放到网上了，试了下classification, retrieval, image2text的在线demo, amazing! http://t.cn/Rvs0Pvj 最重要的是他们的source code以及installation & documentation 也一并公布，超过Rob Fergus学生的Clarifai http://t.cn/8kL993u

斯坦福大学计算机系的大作业，利用深度学习技术（CNN、RNN等）做深度的图像分析，自动描述图片，图片物体识别等等。详细技术报告在这里 http://t.cn/RzLtOfp Project地址：http://t.cn/Rz7F0Fj

### 其他方法

#### NBSVM
[Baselines and bigrams: simple, good sentiment and topic classification]()

我的理解是：把每个抽取得到的feature按照log-count ratio计算了一个weight，再送给标准的svm去做训练。

[Python collections模块实例讲解](http://www.jb51.net/article/48771.htm)

#### RNNLM
[rnnlm lib](http://www.fit.vutbr.cz/~imikolov/rnnlm/)
目前最先进的language model。
cache models (that describe long context information) or class-based models (that improve parameter estimation for short contexts by sharing parameters between similar words)

[Mikolov 的 RNNLM](http://licstar.net/archives/328#s24)，最好的参考资料：博士论文《Statistical Language Models based on Neural Networks》

循环神经网络的最大优势在于，可以真正充分地利用所有上文信息来预测下一个词，而不像前面的其它工作那样，只能开一个 n 个词的窗口，只用前 n 个词来预测下一个词。在训练rnnlm的同时，也可以得到派生物词向量。

word2vec也是mikolov的作品，与rnnlm相比目的不太一样，rnnlm主要是想做语言模型；word2vec的本身目的就是训练词向量，速度会更快一些。

rnnlm是怎么用到二分类上去的? 针对不同的label，训练两个不同的语言模型p+(x\|y=+1)和p-(x\|y=-1)。对于一个testcase x，求解r= p+(x\|y=+1)/p-(x\|y=-1)*p(y=+1)/p(y=-1)，如果r>1，则x属于label(+1)，否则x属于label(-1)。

其预测脚本为：

./rnnlm -rnnlm model-pos -test test-id.txt -debug 0 -nbest > model-pos-score
./rnnlm -rnnlm model-neg -test test-id.txt -debug 0 -nbest > model-neg-score
paste model-pos-score model-neg-score | awk '{print $1 " " $2 " " $1/$2;}' > ../scores/RNNLM-TEST

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/rnnlm_image.png)
![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/rnnlm_formula.png)

####LSTM
http://hlgljmw.com/baijiale/?p/3405569985

原作者写的教程
http://people.idsia.ch/~juergen/lstm/sld001.htm
BPTT很好理解，说是RNN，其实可以理解为每层权重相同的feed forward BP，每层都用时间点上的label来训练，每层的误差都反传，这样就还原为了标准BP网络

然后就会面临BP网络的经典问题，即Exponential Error Decay，误差传4层就传没了！这个东西的具体解释见
Hochreiter,Bengio, Frasconi,(2001) Gradient flow in recurrent nets: The difficulty of learning long-term dependencies （基本看不懂）

http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=3D5F45337EDCE4B9A70877162000D39F?doi=10.1.1.24.7321&rep=rep1&type=pdf
为了解决这个问题而发明了LSTM，目的是为了将这些反传的误差保存起来，它纯粹是为了解决BPTT中的Exponential Error Decay的问题；核心部件（下图红圆）叫做error carousel(误差传送带)，就是一个最简单的RNN积分器；除了这部分以外还有两个网络来控制红色部分的输入输出，分别称为in和out，用来控制红色部分在何时存取
动机（摘自wikipedia）

however, when error values are back-propagated from the output, the error becomes trapped in the memory portion of the block. This is referred to as an "error carousel", which continuously feeds error back to each of the gates until they become trained to cut off the value. Thus, regular backpropagation is effective at training an LSTM block to remember values for very long durations.

#### sentence vector
句向量是通过词向量演化出来的。具体请参考论文[Distributed representations of sentences and documents]()

常见做法是：先利用word2vec训练出每个句子的embedding表示，然后把embedding作为特征送入liblinear。

#### emsemble
ensemble的方法有很多，线性ensemble，指数ensemble。

### 参考论文

1. [Term-weighting approaches in automatic text retrieval，Gerard Salton et.](http://comminfo.rutgers.edu/~muresan/IR/Docs/Articles/ipmSalton1988.pdf)
2. [New term weighting formulas for the vector space method in information retrieval](http://www.sandia.gov/~tgkolda/pubs/pubfiles/ornl-tm-13756.pdf)
3. [A neural probabilistic language model 2003](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
4. [Deep Learning in NLP-词向量和语言模型](http://licstar.net/archives/328)
5. [Recurrent neural network based language models](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
6. Statistical Language Models based on Neural Networks，mikolov博士论文
7. [Rnnlm library](http://www.fit.vutbr.cz/~imikolov/rnnlm/)
8. [A survey of named entity recognition and classification](http://brown.cl.uni-heidelberg.de/~sourjiko/NER_Literatur/survey.pdf)
9. [Deep learning for Chinese word segmentation and POS tagging](http://www.aclweb.org/anthology/D13-1061)
10. [Max-margin tensor neural network for chinese word segmentation](http://aclweb.org/anthology/P14-1028)
11. [Learning distributed representations of concepts](http://www.cogsci.ucsd.edu/~ajyu/Teaching/Cogs202_sp12/Readings/hinton86.pdf)
12. [Care and Feeding of Topic Models: Problems, Diagnostics, and Improvements](http://www.cs.colorado.edu/~jbg/docs/2014_book_chapter_care_and_feeding.pdf)
13. [LightLda](http://arxiv.org/abs/1412.1576)
14. [word2vec](https://code.google.com/p/word2vec/)
15. [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)
16. [Deep Learning实战之word2vec](http://techblog.youdao.com/?p=915)
17. [word2vec中的数学原理详解](http://suanfazu.com/t/word2vec-zhong-de-shu-xue-yuan-li-xiang-jie-duo-tu-wifixia-yue-du/178) [出处2](http://blog.csdn.net/itplus/article/details/37969519)
18. [斯坦福课程-语言模型](http://52opencourse.com/111/%E6%96%AF%E5%9D%A6%E7%A6%8F%E5%A4%A7%E5%AD%A6%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E7%AC%AC%E5%9B%9B%E8%AF%BE-%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%EF%BC%88language-modeling%EF%BC%89)
19. [Translating Videos to Natural Language Using Deep Recurrent Neural Networks](http://arxiv.org/abs/1412.4729)
20. [Distributed Representations of Sentences and Documents](http://arxiv.org/pdf/1405.4053v2.pdf)
21. [Convolutional Neural Networks卷积神经网络](http://blog.csdn.net/zouxy09/article/details/8781543)
22. [A New, Deep-Learning Take on Image Recognition](http://research.microsoft.com/en-us/news/features/spp-102914.aspx)
23. [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](http://arxiv.org/pdf/1406.4729v1.pdf)
24. [A Deep Learning Tutorial: From Perceptrons to Deep Networks](http://www.toptal.com/machine-learning/an-introduction-to-deep-learning-from-perceptrons-to-deep-networks)
25. [Deep Learning for Computer Vision](http://cs.nyu.edu/~fergus/presentations/nips2013_final.pdf)
26. [Zero-shot leanring by convex combination of semantic embeddings](http://arxiv.org/pdf/1312.5650.pdf)
27. [Sequence to sequence learning with neural network](http://arxiv.org/pdf/1409.3215v3.pdf)
28. [Exploting similarities among language for machine translation](http://arxiv.org/pdf/1309.4168.pdf)
29. Grammar as Foreign Language Oriol Vinyals, Lukasz Kaiser, Terry Koo, Slav Petrov, Ilya Sutskever, Geoffrey Hinton, arXiv 2014
30. [Deep Semantic Embedding](http://ceur-ws.org/Vol-1204/papers/paper_4.pdf)
31. 张家俊. DNN Applications in NLP
32. [Deep learning for natural language processing and machine translation](http://cl.naist.jp/~kevinduh/notes/cwmt14tutorial.pdf)
33. [Distributed Representations for Semantic Matching]()
34. distributed_representation_nlp
35. Deep Visual-Semantic Alignments for Generating Image Descriptions
36. [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/pdf/1408.5882v2.pdf)
37. [Senna](http://ml.nec-labs.com/senna)
38. [ImageNet Large Scale Visual Recognition Challenge](http://arxiv.org/pdf/1409.0575v1.pdf)
39. Krizhevsky A, Sutskever I, Hinton G E. ImageNet Classification with Deep Convolutional Neural Networks
40. [Gradient-Based Learning Applied to Document Recognition](http://turing.iimas.unam.mx/~elena/CompVis/Lecun98.pdf)
41. Effetive use of word order for text categorization with convolutional neural network，Rie Johnson
42. [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](http://arxiv.org/pdf/1406.1078.pdf)
43. [Show and Tell: A Neural Image Caption Generator](http://arxiv.org/pdf/1411.4555v1.pdf)
44. [Deep Image: Scaling up Image Recognition](http://arxiv.org/ftp/arxiv/papers/1501/1501.02876.pdf)
45. Large-Scale High-Precision Topic Modeling on Twitter
46. A. Krizhevsky. One weird trick for parallelizing convolutional neural networks. arXiv:1404.5997, 2014
47. [A Brief Overview of Deep Learning](http://yyue.blogspot.com/2015/01/a-brief-overview-of-deep-learning.html)
48. Going deeper with convolutions. Christian Szegedy. Google Inc. [阅读笔记](http://www.gageet.com/2014/09203.php)
49. Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling
50. [Semi-Supervised Learning Tutorial](http://pages.cs.wisc.edu/~jerryzhu/pub/sslicml07.pdf)
51. http://www.zhihu.com/question/24904450
52. [LONG SHORT-TERM MEMORY BASED RECURRENT NEURAL NETWORK ARCHITECTURES FOR LARGE VOCABULARY SPEECH RECOGNITION](http://arxiv.org/pdf/1402.1128.pdf)
53. [LSTM Neural Networks for Language Modeling](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.248.4448&rep=rep1&type=pdf)
54. [LONG SHORT-TERM MEMORY](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
55. Bengio, Y., Simard, P., Frasconi, P., “Learning long-term dependencies with gradient descent is difficult” IEEE Transactions on Neural Networks 5 (1994), pp. 157–166
56. [AliasLDA](http://www.sravi.org/pubs/fastlda-kdd2014.pdf)
57. [Gibbs sampling for the uninitiated](www.umiacs.umd.edu/~resnik/pubs/LAMP-TR-153.pdf)
58. [Learning classifiers from only positive and unlabeled data](http://www.eecs.tufts.edu/~noto/pub/kdd08/elkan.kdd08.poster.pdf)
59. [TF-ICF: A New Term Weighting Scheme for Clustering Dynamic Data Streams](http://cda.ornl.gov/publications/ICMLA06.pdf)
60. [LDA数学八卦](http://www.flickering.cn/%E6%A6%82%E7%8E%87%E7%BB%9F%E8%AE%A1/2014/06/%E3%80%90lda%E6%95%B0%E5%AD%A6%E5%85%AB%E5%8D%A6%E3%80%91%E7%A5%9E%E5%A5%87%E7%9A%84gamma%E5%87%BD%E6%95%B0/)
61. [Chinese Word Segmentation and Named Entity Recognition Based on Conditional Random Fields Models](http://www.aclweb.org/anthology/W06-0132)
62. [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](http://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)
63. [Chinese Segmentation and New Word Detection using Conditional Random Fields](http://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1091&context=cs_faculty_pubs)
64. [Gregor Heinrich. Parameter estimation for text analysis](http://www.arbylon.net/publications/text-est.pdf)
65. [Peacock：大规模主题模型及其在腾讯业务中的应用](http://km.oa.com/group/14352/articles/show/213192)
66. L. Yao, D. Mimno, and A. McCallum. Efficient methods for topic model inference on streaming document collections. In KDD, 2009.
67. [David Newman. Distributed Algorithms for Topic Models](http://www.jmlr.org/papers/volume10/newman09a/newman09a.pdf)
68. [Xuemin. LDA工程实践之算法篇](http://www.flickering.cn/nlp/2014/07/lda工程实践之算法篇-1算法实现正确性验证/)
69. [Brian Lott. Survey of Keyword Extraction Techniques](http://www.cs.unm.edu/~pdevineni/papers/Lott.pdf)
70. Yi Wang, Xuemin Zhao, Zhenlong Sun, Hao Yan, Lifeng Wang, Zhihui Jin, Liubin Wang, Yang Gao, Ching Law, and Jia Zeng. Peacock: Learning Long-Tail Topic Features for Industrial Applications. TIST’2015.
71. [刘知远. 基于文档主题结构的关键词抽取方法研究](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/phd_thesis.pdf)
72. [Hinton. Reducing the Dimensionality of Data with Neural Networks](http://www.cs.toronto.edu/~hinton/science.pdf)
73. [Samaneh Moghaddam. On the design of LDA models for aspect-based opinion mining](http://dl.acm.org/citation.cfm?id=2396863)；
74. The FLDA model for aspect-based opinion mining: addressing the cold start problem
75. [Ross Girshick et. Rich feature hierarchies for accurate object detection and semantic segmentation](http://www.cs.berkeley.edu/~rbg/papers/r-cnn-cvpr.pdf)
76. J. Uijlings, K. van de Sande, T. Gevers, and A. Smeulders. Selective search for object recognition. IJCV, 2013.
77. [Baidu/UCLA: Explain Images with Multimodal Recurrent Neural Networks](http://arxiv.org/abs/1410.1090)
78. [Toronto: Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models](http://arxiv.org/abs/1411.2539)
79. [Berkeley: Long-term Recurrent Convolutional Networks for Visual Recognition and Description](http://arxiv.org/abs/1411.4389)
80. [Xinlei Chen et. Learning a Recurrent Visual Representation for Image Caption Generation](http://arxiv.org/abs/1411.5654)
81. [Hao Fang et. From Captions to Visual Concepts and Back](http://arxiv.org/pdf/1411.4952v2)
82. [Modeling Documents with a Deep Boltzmann Machine](http://www.cs.toronto.edu/~nitish/uai13.pdf)
83. [A Deep Dive into Recurrent Neural Nets](http://nikhilbuduma.com/2015/01/11/a-deep-dive-into-recurrent-neural-networks/)
84. [Xiang zhang et. Text Understanding from Scratch](http://arxiv.org/abs/1502.01710?utm_source=dlvr.it&utm_medium=tumblr)

word2vec的其他参考文章：

- http://blog.csdn.net/mytestmy/article/details/26961315
- http://blog.csdn.net/mytestmy/article/details/26969149
- www.zhihu.com/question/21661274/answer/19331979
- http://suanfazu.com/t/wen-ben-shen-du-biao-shi-mo-xing-word2vec/258
- [word2vec在事件挖掘中的运用](http://blog.csdn.net/shuishiman/article/details/20769437#1536434-tsina-1-26292-66a1f5d8f89e9ad52626f6f40fdeadaa)

Zero-shot：

- [zero-shot leanring by convex combination of semantic embeddings]()
- [distributed representations of sentences and documents]()

machine translation:

- [sequence to sequence learning with neural network]()
- [exploting similarities among language for machine translation]()

SVD:

- [Fast Randomized SVD](https://research.facebook.com/blog/294071574113354/fast-randomized-svd/)

相似度计算：

- [chinese-word-similarity](https://github.com/memect/hao/blob/master/awesome/chinese-word-similarity.md)

中文分词参考文献：

- [白话中文分词之HMM模型](http://yanyiwu.com/work/2014/04/07/hmm-segment-xiangjie.html)
- [浅谈中文分词](http://www.isnowfy.com/introduction-to-chinese-segmentation/)
- [分词km](http://km.oa.com/news/post/21560)
- [52nlp中文分词](http://www.52nlp.cn/category/word-segmentation)
- [Deep Learning 在中文分词和词性标注任务中的应用](http://blog.csdn.net/itplus/article/details/13616045)
- [利用 word2vec 训练的字向量进行中文分词](http://blog.csdn.net/itplus/article/details/17122431)
- [分词原理1](http://blog.sina.com.cn/s/blog_7eb42b5a0100vf8l.html)
- [分词原理2](http://blog.sina.com.cn/s/blog_6876a34b0100uq49.html)
- [分词原理3](http://www.cnblogs.com/flish/archive/2011/08/08/2131031.html)

半监督学习：

- [半监督学习1](http://blog.csdn.net/ice110956/article/details/13775071)
- [半监督学习2](http://www.cnblogs.com/liqizhou/archive/2012/05/11/2496155.html)
- [Semi-Supervised Learning Tutorial](http://pages.cs.wisc.edu/~jerryzhu/pub/sslicml07.pdf)

大规模机器学习：

- [Muli：Scaling Distributed Machine Learning with the Parameter Server](http://www.cs.cmu.edu/~muli/file/pdl14_talk.pdf)

图片搜索：

- [图片搜索的原理](http://lusongsong.com/info/post/155.html)
- [相似图片搜索的原理](http://www.ruanyifeng.com/blog/2011/07/principle_of_similar_image_search.html)

卷积：

- [卷积的物理意义](http://www.zhihu.com/question/21686447)

RNN，CNN paper整理：

- [Deep Learning方向的paper整理](http://hi.baidu.com/chb_seaok/item/6307c0d0363170e73cc2cb65)

R-CNN:

- [rcnn1](http://blog.csdn.net/chenriwei2/article/details/41725871)
- [rcnn2](http://blog.csdn.net/chenriwei2/article/details/38110387)

Image Understand：

- [image-captioning](https://pdollar.wordpress.com/2015/01/21/image-captioning/)
- [image-captioning2](http://blogs.technet.com/b/machinelearning/archive/2014/11/18/rapid-progress-in-automatic-image-captioning.aspx)

语义分析 新文献:

- Learning Image Embeddings using Convolutional Neural Networks for Improved Multi-Modal Semantics
- A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval
- Unsupervised Learning of Word Semantic Embedding using the Deep Structured Semantic Model
- Semantics of Visual Discrimination
