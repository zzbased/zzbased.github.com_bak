---
layout: post
title: "语义分析的一些方法"
description: ""
category:
tags: [machine learning, nlp]
---
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

author: vincentyao@tencent.com

语义分析，本文指运用各种机器学习方法，挖掘与学习文本、图片等的深层次概念。wikipedia上的解释：In machine learning, semantic analysis of a corpus is the task of building structures that approximate concepts from a large set of documents(or images)。

工作这几年，陆陆续续实践过一些项目，有搜索广告，社交广告，微博广告，品牌广告，内容广告等。要使我们广告平台效益最大化，首先需要理解用户，Context(将展示广告的上下文)和广告，才能将最合适的广告展示给用户。而这其中，就离不开对用户，对上下文，对广告的语义分析，由此催生了一些子项目，例如文本语义分析，图片语义理解，语义索引，短串语义关联，用户广告语义匹配等。

接下来我将写一写我所认识的语义分析的一些方法，虽说我们在做的时候，效果导向居多，方法理论理解也许并不深入，不过权当个人知识点总结，有任何不当之处请指正，谢谢。

本文主要由以下四部分组成：文本基本处理，文本语义分析，图片语义分析，语义分析小结。先讲述文本处理的基本方法，这构成了语义分析的基础。接着分文本和图片两节讲述各自语义分析的一些方法，值得注意的是，虽说分为两节，但文本和图片在语义分析方法上有很多共通与关联。最后我们简单介绍下语义分析在广点通"用户广告匹配"上的应用，并展望一下未来的语义分析方法。

### 1 文本基本处理

在讲文本语义分析之前，我们先说下文本基本处理，因为它构成了语义分析的基础。而文本处理有很多方面，考虑到本文主题，这里只介绍中文分词以及Term Weighting。

#### 1.1 中文分词
拿到一段文本后，通常情况下，首先要做分词。分词的方法一般有如下几种：

- 基于字符串匹配的分词方法。此方法按照不同的扫描方式，逐个查找词库进行分词。根据扫描方式可细分为：正向最大匹配，反向最大匹配，双向最大匹配，最小切分(即最短路径)；总之就是各种不同的启发规则。

- 全切分方法。它首先切分出与词库匹配的所有可能的词，再运用统计语言模型决定最优的切分结果。它的优点在于可以解决分词中的歧义问题。下图是一个示例，对于文本串"南京市长江大桥"，首先进行词条检索(一般用Trie存储)，找到匹配的所有词条（南京，市，长江，大桥，南京市，长江大桥，市长，江大桥，江大，桥），以词网格(word lattices)形式表示，接着做路径搜索，基于统计语言模型(例如n-gram)[18]找到最优路径，最后可能还需要命名实体识别。下图中"南京市 长江 大桥"的语言模型得分，即P(南京市，长江，大桥)最高，则为最优切分。

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/rnnlm1.png)

	图1. "南京市长江大桥"语言模型得分

- 由字构词的分词方法。可以理解为字的分类问题，也就是自然语言处理中的sequence labeling问题，通常做法里利用HMM，MAXENT，MEMM，CRF等预测文本串每个字的tag[62]，譬如B，E，I，S，这四个tag分别表示：beginning, ending, inside, single，也就是一个词的开始，结束，中间，以及单个字的词。 例如"南京市长江大桥"的标注结果可能为："南(B)京(I)市(E)长(B)江(E)大(B)桥(E)"

	由于CRF既可以像最大熵模型一样加各种领域feature，又避免了HMM的齐次马尔科夫假设，所以基于CRF的分词目前是效果最好的，具体请参考文献[61,62,63]。

	除了HMM，CRF等模型，分词也可以基于深度学习方法来做，如文献[9][10]所介绍，也取得了state-of-the-art的结果。

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/word_segmentation.png)

	图2. 基于深度学习的中文分词

	上图是一个基于深度学习的分词示例图。我们从上往下看，首先对每一个字进行Lookup Table，映射到一个固定长度的特征向量(这里可以利用词向量，boundary entropy，accessor variety等)；接着经过一个标准的神经网络，分别是linear，sigmoid，linear层，对于每个字，预测该字属于B,E,I,S的概率；最后输出是一个矩阵，矩阵的行是B,E,I,S 4个tag，利用viterbi算法就可以完成标注推断，从而得到分词结果。

一个文本串除了分词，还需要做词性标注，命名实体识别，新词发现等。通常有两种方案，一种是pipeline approaches，就是先分词，再做词性标注；另一种是joint approaches，就是把这些任务用一个模型来完成。有兴趣可以参考文献[9][62]等。

一般而言，方法一和方法二在工业界用得比较多，方法三因为采用复杂的模型，虽准确率相对高，但耗时较大。在腾讯内部采用的是方法二。

#### 1.2 语言模型
前面在讲"全切分分词"方法时，提到了语言模型，并且通过语言模型，还可以引出词向量，所以这里把语言模型简单阐述一下。

语言模型是用来计算一个句子产生概率的概率模型，即\\(P(w_1,w_2,w_3...w_m)\\)，m表示词的总个数。根据贝叶斯公式：\\(P(w_1,w_2,w_3 ... w_m) = P(w_1)P(w_2\|w_1)P(w_3\|w_1,w_2) ... P(w_m\|w_1,w_2 ... w_{m-1})\\)。

最简单的语言模型是N-Gram，它利用马尔科夫假设，认为句子中每个单词只与其前n-1个单词有关，即假设产生w_m这个词的条件概率只依赖于前n-1个词，则有 `\(P(w_m \|w_1,w_2...w_{m-1}) = P(w_m \|w_{m-n+1},w_{m-n+2} ... w_{m-1})\)`。

其中n越大，模型可区别性越强，n越小，模型可靠性越高。

N-Gram语言模型简单有效，但是它只考虑了词的位置关系，没有考虑词之间的相似度，词法和词语义，并且还存在数据稀疏的问题，所以后来，又逐渐提出更多的语言模型，例如Class-based ngram model，topic-based ngram model，cache-based ngram model，skipping ngram model，指数语言模型（最大熵模型，条件随机域模型）等。若想了解更多请参考文章[18]。

最近，随着深度学习的兴起，神经网络语言模型也变得火热[4]。用神经网络训练语言模型的经典之作，要数Bengio等人发表的《A Neural Probabilistic Language Model》[3]，它也是基于n-gram的，首先将每个单词 \\(w_{m-n+1},w_{m-n+2} ... w_{m-1}\\) 映射到词向量空间，再把各个单词的词向量组合成一个更大的向量作为神经网络输入，输出是\\(P(w_m)\\)。本文将此模型简称为ffnnlm（Feed-forward Neural Net Language Model）。ffnnlm解决了传统n-gram的两个缺陷：(1)词语之间的相似性可以通过词向量来体现；(2)自带平滑功能。文献[3]不仅提出神经网络语言模型，还顺带引出了词向量，关于词向量，后文将再细述。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/ffnnlm.png)

图3. 基于神经网络的语言模型

从最新文献看，目前state-of-the-art语言模型应该是基于循环神经网络(recurrent neural network)的语言模型，简称rnnlm[5][6]。循环神经网络相比于传统前馈神经网络，其特点是：可以存在有向环，将上一次的输出作为本次的输入。而rnnlm和ffnnlm的最大区别是：ffnnmm要求输入的上下文是固定长度的，也就是说n-gram中的 n 要求是个固定值，而rnnlm不限制上下文的长度，可以真正充分地利用所有上文信息来预测下一个词，本次预测的中间隐层信息(例如下图中的context信息)可以在下一次预测里循环使用。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/simple_rnn.png)

图4. 基于simple RNN(time-delay neural network)的语言模型

如上图所示，这是一个最简单的rnnlm，神经网络分为三层，第一层是输入层，第二层是隐藏层(也叫context层)，第三层输出层。
假设当前是t时刻，则分三步来预测\\(P(w_m)\\)：

- 单词\\(w_{m-1}\\)映射到词向量，记作input(t)
- 连接上一次训练的隐藏层context(t-1)，经过sigmoid function，生成当前t时刻的context(t)
- 利用softmax function，预测\\(P(w_m)\\)

参考文献[7]中列出了一个rnnlm的library，其代码紧凑。利用它训练中文语言模型将很简单，上面"南京市 长江 大桥"就是rnnlm的预测结果。

基于RNN的language model利用BPTT(BackPropagation through time)算法比较难于训练，原因就是深度神经网络里比较普遍的vanishing gradient问题[55]（在RNN里，梯度计算随时间成指数倍增长或衰减，称之为Exponential Error Decay）。所以后来又提出基于LSTM(Long short term memory)的language model，LSTM也是一种RNN网络，关于LSTM的详细介绍请参考文献[54,49,52]。LSTM通过网络结构的修改，从而避免vanishing gradient问题，具体分析请参考文献[83]。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/lstm_unit.png)

图5. LSTM memory cell

如上图所示，是一个LSTM unit。如果是传统的神经网络unit，output activation bi = activation_function(ai)，但LSTM unit的计算相对就复杂些了，它保存了该神经元上一次计算的结果，通过input gate，output gate，forget gate来计算输出，具体过程请参考文献[53，54]。

#### 1.3 Term Weighting
##### Term重要性
对文本分词后，接下来需要对分词后的每个term计算一个权重，重要的term应该给与更高的权重。举例来说，"什么产品对减肥帮助最大？"的term weighting结果可能是: "什么 0.1，产品 0.5，对 0.1，减肥 0.8，帮助 0.3，最大 0.2"。Term weighting在文本检索，文本相关性，核心词提取等任务中都有重要作用。

- Term weighting的打分公式一般由三部分组成：local，global和normalization [1,2]。即
\\(TermWeight=L_{i,j} G_i N_j\\)。\\(L_{i,j}\\)是term i在document j中的local weight，\\(G_i\\)是term i的global weight，\\(N_j\\)是document j的归一化因子。
常见的local，global，normalization weight公式[2]有：

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/local_weight.png)

	图6. Local weight formulas

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/global_weight.png)

	图7. Global weight formulas

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/normlization_weight.png)

	图8. Normalization factors

	Tf-Idf是一种最常见的term weighting方法。在上面的公式体系里，Tf-Idf的local weight是FREQ，glocal weight是IDFB，normalization是None。tf是词频，表示这个词出现的次数。df是文档频率，表示这个词在多少个文档中出现。idf则是逆文档频率，idf=log(TD/df)，TD表示总文档数。tf-idf在很多场合都很有效，但缺点也比较明显，以"词频"度量重要性，不够全面，譬如在搜索广告的关键词匹配时就不够用。

	除了TF-IDF外，还有很多其他term weighting方法，例如Okapi，MI，LTU，ATC，TF-ICF[59]等。通过local，global，normalization各种公式的组合，可以生成不同的term weighting计算方法。不过上面这些方法都是无监督计算方法，有一定程度的通用性，但在一些特定场景里显得不够灵活，不够准确，所以可以基于有监督机器学习方法来拟合term weighting结果。

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/okapi.png)

	图9. Okapi计算公式

- 利用有监督机器学习方法来预测weight。这里类似于机器学习的分类任务，对于文本串的每个term，预测一个[0,1]的得分，得分越大则term重要性越高。

	既然是有监督学习，那么就需要训练数据。如果采用人工标注的话，极大耗费人力，所以可以采用训练数据自提取的方法，利用程序从搜索日志里自动挖掘。从海量日志数据里提取隐含的用户对于term重要性的标注，得到的训练数据将综合亿级用户的"标注结果"，覆盖面更广，且来自于真实搜索数据，训练结果与标注的目标集分布接近，训练数据更精确。下面列举三种方法(除此外，还有更多可以利用的方法)：

	- 从搜索session数据里提取训练数据，用户在一个检索会话中的检索核心意图是不变的，提取出核心意图所对应的term，其重要性就高。
	- 从历史短串关系资源库里提取训练数据，短串扩展关系中，一个term出现的次数越多，则越重要。
	- 从搜索广告点击日志里提取训练数据，query与bidword共有term的点击率越高，它在query中的重要程度就越高。

	通过上面的方法，可以提取到大量质量不错的训练数据（数十亿级别的数据，这其中可能有部分样本不准确，但在如此大规模数据情况下，绝大部分样本都是准确的）。

	有了训练数据，接下来提取特征，基于逻辑回归模型来预测文本串中每个term的重要性。所提取的特征包括：

	- term的自解释特征，例如term专名类型，term词性，term idf，位置特征，term的长度等；
	- term与文本串的交叉特征，例如term与文本串中其他term的字面交叉特征，term转移到文本串中其他term的转移概率特征，term的文本分类、topic与文本串的文本分类、topic的交叉特征等。

##### 核心词、关键词提取
- 短文本串的核心词提取。对短文本串分词后，利用上面介绍的term weighting方法，获取term weight后，取一定的阈值，就可以提取出短文本串的核心词。

- 长文本串(譬如web page)的关键词提取。这里简单介绍几种方法。想了解更多，请参考文献[69]。
	- 采用基于规则的方法。考虑到位置特征，网页特征等。
	- 基于广告主购买的bidword和高频query建立多模式匹配树，在长文本串中进行全字匹配找出候选关键词，再结合关键词weight，以及某些规则找出优质的关键词。
	- 类似于有监督的term weighting方法，也可以训练关键词weighting的模型。
	- 基于文档主题结构的关键词抽取，具体可以参考文献[71]。

### 2 文本语义分析
前面讲到一些文本基本处理方法。一个文本串，对其进行分词和重要性打分后（当然还有更多的文本处理任务），就可以开始更高层的语义分析任务。

#### 2.1 Topic Model
首先介绍主题模型。说到主题模型，第一时间会想到pLSA，NMF，LDA。关于这几个目前业界最常用的主题模型，已经有相当多的介绍了，譬如文献[60，64]。在这里，主要想聊一下主题模型的应用以及最新进展(考虑到LDA是pLSA的generalization，所以下面只介绍LDA)。

##### LDA训练算法简单介绍
LDA的推导这里略过不讲，具体请参考文献[64]。下面我们主要看一下怎么训练LDA。

在Blei的原始论文中，使用variational inference和EM算法进行LDA推断(与pLSA的推断过程类似，E-step采用variational inference)，但EM算法可能推导出局部最优解，且相对复杂。目前常用的方法是基于gibbs sampling来做[57]。

- Step1: 随机初始化每个词的topic，并统计两个频率计数矩阵：Doc-Topic 计数矩阵N(t,d)，描述每个文档中的主题频率分布；Word-Topic 计数矩阵N(w,t)，表示每个主题下词的频率分布。
- Step2: 遍历训练语料，按照概率公式(下图所示)重新采样每个词所对应的topic, 更新N(t,d)和N(w,t)的计数。
- Step3: 重复 step2，直到模型收敛。

对文档d中词w的主题z进行重新采样的公式有非常明确的物理意义，表示为P(w\|z)P(z\|d)，直观的表示为一个“路径选择”的过程。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/lda_sampling.png)

图10. gibbs sampling过程图

以上描述过程具体请参考文献[65]。

对于LDA模型的更多理论介绍，譬如如何实现正确性验证，请参考文献[68]，而关于LDA模型改进，请参考Newman团队的最新文章《Care and Feeding of Topic Models》[12]。

##### 主题模型的应用点
- 在广点通内部，主题模型已经在很多方面都得到成功应用[65]，譬如文本分类特征，相关性计算，ctr预估，精确广告定向，矩阵分解等。具体来说，基于主题模型，可以计算出文本，用户的topic分布，将其当作pctr，relevance的特征，还可以将其当作一种矩阵分解的方法，用于降维，推荐等。不过在我们以往的成功运用中，topic模型比较适合用做某些机器学习任务的特征，而不适合作为一种独立的方法去解决某种特定的问题，例如触发，分类。Blei是这样评价lda的：it can easily be used as a module in more complicated models for more complicated goals。

- 为什么topic model不适合作为一种独立的方法去解决某种特定的问题(例如分类，触发等)。
	- 个人总结，主要原因是lda模型可控性可解释性相对比较差：对于每个topic，不能用很明确的语义归纳出这个topic在讲什么；重新训练一遍lda模型，每个topic id所对应的语义可能发生了变化；有些topic的准确性比较好，有些比较差，而对于比较差的topic，没有特别好的针对性的方法去优化它；
	- 另外一个就是topic之间的重复，特别是在topic数目比较多的情况，重复几乎是不可避免的，当时益总在开发peacock的时候，deduplicate topic就是一个很重要的任务。如果多个topic描述的意思一致时，用topic id来做检索触发，效果大半是不好的，后来我们也尝试用topic word来做，但依旧不够理想。

##### 主题模型最新进展
首先主题模型自PLSA, LDA后，又提出了很多变体，譬如HDP。LDA的topic number是预先设定的，而HDP的topic number是不固定，而是从训练数据中学习得到的，这在很多场景是有用的，具体参考[hdp vs lda](http://datascience.stackexchange.com/questions/128/latent-dirichlet-allocation-vs-hierarchical-dirichlet-process)。想了解更多LDA模型的升级，请参考文献[73,74]。

深度学习方面，Geoff Hinton及其学生用Deep Boltzmann Machine研究出了类似LDA的隐变量文本模型[82]，文章称其抽取的特征在文本检索与文本分类上的结果比LDA好。[heavenfireray](http://weibo.com/u/2387864597)在其微博评论道：lda结构是word-hidden topic。类lda结构假设在topic下产生每个word是条件独立而且参数相同。这种假设导致参数更匹配长文而非短文。该文章提出word-hidden topic-hidden word，其实是(word,hidden word)-hidden topic，增加的hidden word平衡了参数对短文的适配，在分类文章数量的度量上更好很自然。

其次，随着目前互联网的数据规模的逐渐增加，大规模并行PLSA，LDA训练将是主旋律。大规模主题模型训练，除了从系统架构上进行优化外，更关键的，还需要在算法本身上做升级。variational方法不太适合并行化，且速度相对也比较慢，这里我们着重看sampling-base inference。

- collapsed Gibbs sampler[57]：O(K)复杂度，K表示topic的总个数。
- SparseLDA[66]：算法复杂度为O(Kd + Kw)，Kd表示文档d所包含的topic个数，Kw表示词w所属的topic个数，考虑到一个文档所包含的topic和一个词所属的topic个数是有限的，肯定远小于K，所以相比于collapsed Gibbs，复杂度已有较大的下降。
- AliasLDA[56]：利用alias table和Metropolis-Hastings，将词这个维度的采样复杂度降至O(1)。所以算法总复杂度为O(Kd)。
- Metropolis-Hastings sampler[13]：复杂度降至O(1)。这里不做分析了，具体请参考文献[13]

##### 主题模型并行化
在文献[67]中，Newman团队提出了LDA算法的并行化版本Approximate distributed-LDA，如下图所示：

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/ad_lda.png)

图11. AD-LDA算法

在原始gibbs sampling算法里，N(w,t)这个矩阵的更新是串行的，但是研究发现，考虑到N(w,t)矩阵在迭代过程中，相对变化较小，多个worker独立更新N(w,t)，在一轮迭代结束后再根据多个worker的本地更新合并到全局更新N(w,t)，算法依旧可以收敛[67]。

那么，主题模型的并行化(不仅仅是主题模型，其实是绝大部分机器学习算法)，主要可以从两个角度来说明：数据并行和模型并行。

- 数据并行。这个角度相对比较直观，譬如对于LDA模型，可以将训练数据按照worker数目切分为M片(M为worker数)，每个worker保存一份全局的N(w,t)矩阵，在一轮迭代里，各个worker独立计算，迭代结束后，合并各个worker的本地更新。这个思路可以借用目前通用的并行计算框架，譬如Spark，Hadoop，Graphlab等来实现。
- 模型并行。考虑到矩阵N(w,t)在大规模主题模型中相当巨大，单机内存不可能存下。所以直观的想法，可以将N(w,t)也切分成多个分片。N(w,t)可以考虑使用全局的parameter server来存储，也可以考虑存储在不同worker上，利用MPI AllReduce来通信。

数据与模型并行，可以形象的描述为一个棋盘。棋盘的行按照数据划分，棋盘的列按照模型划分。LDA的并行化，就是通过这样的切分，将原本巨大的，不可能在单机存储的矩阵切分到不同的机器，使每台机器都能够将参数存储在内存。再接着，各个worker相对独立计算，计算的过程中不时按照某些策略同步模型数据。

最近几年里，关于LDA并行化已有相当多的开源实现，譬如：

- PLDA，[PLDA+](https://code.google.com/p/plda/)
- [Yahoo LDA](https://github.com/sudar/Yahoo_LDA)
- [Parameter server](https://github.com/mli/parameter_server)

最近的并行LDA实现Peacock[70,65]和LigthLda[13]没有开源()，但我们可以从其论文一窥究竟，总体来说，并行化的大体思路是一致的。譬如LightLDA[13]，下图是实现架构框图，它将训练数据切分成多个Block，模型通过parameter server来同步，每个data block，类似于sliding windows，在计算完V1的采样后，才会去计算V2的采样(下图中V1,V2,V3表示word空间的划分，即模型的划分)。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/parallelism_lda.png)

图12. LightLda并行结构图

#### 2.2 词向量，句向量
##### 词向量是什么
在文本分析的vector space model中，是用向量来描述一个词的，譬如最常见的One-hot representation。One-hot representation方法的一个明显的缺点是，词与词之间没有建立关联。在深度学习中，一般用Distributed Representation来描述一个词，常被称为"Word Representation"或"Word Embedding"，也就是我们俗称的"词向量"。

词向量起源于hinton在1986年的论文[11]，后来在Bengio的ffnnlm论文[3]中，被发扬光大，但它真正被我们所熟知，应该是word2vec[14]的开源。在ffnnlm中，词向量是训练语言模型的一个副产品，不过在word2vec里，是专门来训练词向量，所以word2vec相比于ffnnlm的区别主要体现在：

- 模型更加简单，去掉了ffnnlm中的隐藏层，并去掉了输入层跳过隐藏层直接到输出层的连接。
- 训练语言模型是利用第m个词的前n个词预测第m个词，而训练词向量是用其前后各n个词来预测第m个词，这样做真正利用了上下文来预测，如下图所示。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/word2vec.png)

图13. word2vec的训练算法

上图是word2vec的两种训练算法：CBOW(continuous bag-of-words)和Skip-gram。在cbow方法里，训练目标是给定一个word的context，预测word的概率；在skip-gram方法里，训练目标则是给定一个word，预测word的context的概率。

关于word2vec，在算法上还有较多可以学习的地方，例如利用huffman编码做层次softmax，negative sampling，工程上也有很多trick，具体请参考文章[16][17]。

##### 词向量的应用
词向量的应用点：

- 可以挖掘词之间的关系，譬如同义词。
- 可以将词向量作为特征应用到其他机器学习任务中，例如作为文本分类的feature，Ronan collobert在Senna[37]中将词向量用于POS, CHK, NER等任务。
- 用于机器翻译[28]。分别训练两种语言的词向量，再通过词向量空间中的矩阵变换，将一种语言转变成另一种语言。
- word analogy，即已知a之于b犹如c之于d，现在给出 a、b、c，C(a)-C(b)+C(c)约等于C(d)，C(*)表示词向量。可以利用这个特性，提取词语之间的层次关系。
- Connecting Images and Sentences，image understanding。例如文献，DeViSE: A deep visual-semantic em-bedding model。
- Entity completion in Incomplete Knowledge bases or ontologies，即relational extraction。Reasoning with neural tensor net- works for knowledge base completion。
- more word2vec applications，点击[link1](http://www.quora.com/Do-you-know-any-interesting-applications-using-distributed-representations-of-words-obtained-from-NNLM-eg-word2vec)，[link2](https://www.quora.com/What-are-some-interesting-Word2Vec-results)

除了产生词向量，word2vec还有很多其他应用领域，对此我们需要把握两个概念：doc和word。在词向量训练中，doc指的是一篇篇文章，word就是文章中的词。

- 假设我们将一簇簇相似的用户作为doc（譬如QQ群），将单个用户作为word，我们则可以训练user distributed representation，可以借此挖掘相似用户。
- 假设我们将一个个query session作为doc，将query作为word，我们则可以训练query distributed representation，挖掘相似query。

##### 句向量

分析完word distributed representation，我们也许会问，phrase，sentence是否也有其distributed representation。最直观的思路，对于phrase和sentence，我们将组成它们的所有word对应的词向量加起来，作为短语向量，句向量。在参考文献[34]中，验证了将词向量加起来的确是一个有效的方法，但事实上还有更好的做法。

Le和Mikolov在文章《Distributed Representations of Sentences and Documents》[20]里介绍了sentence vector，这里我们也做下简要分析。

- 先看c-bow方法，相比于word2vec的c-bow模型，区别点有：

	- 训练过程中新增了paragraph id，即训练语料中每个句子都有一个唯一的id。paragraph id和普通的word一样，也是先映射成一个向量，即paragraph vector。paragraph vector与word vector的维数虽一样，但是来自于两个不同的向量空间。在之后的计算里，paragraph vector和word vector累加或者连接起来，作为输出层softmax的输入。在一个句子或者文档的训练过程中，paragraph id保持不变，共享着同一个paragraph vector，相当于每次在预测单词的概率时，都利用了整个句子的语义。
	- 在预测阶段，给待预测的句子新分配一个paragraph id，词向量和输出层softmax的参数保持训练阶段得到的参数不变，重新利用梯度下降训练待预测的句子。待收敛后，即得到待预测句子的paragraph vector。

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/sentence2vec0.png)

	图14. sentence2vec cbow算法

- sentence2vec相比于word2vec的skip-gram模型，区别点为：在sentence2vec里，输入都是paragraph vector，输出是该paragraph中随机抽样的词。

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/sentence2vec1.png)

	图15. sentence2vec skip-gram算法

下面是sentence2vec的结果示例。先利用中文sentence语料训练句向量，然后通过计算句向量之间的cosine值，得到最相似的句子。可以看到句向量在对句子的语义表征上还是相当惊叹的。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/sentence2vec4.png)

图16. sentence2vec 结果示例

##### 词向量的改进
- 学习词向量的方法主要分为：Global matrix factorization和Shallow Window-Based。Global matrix factorization方法主要利用了全局词共现，例如LSA；Shallow Window-Based方法则主要基于local context window，即局部词共现，word2vec是其中的代表；Jeffrey Pennington在word2vec之后提出了[GloVe](http://nlp.stanford.edu/projects/glove/)，它声称结合了上述两种方法，提升了词向量的学习效果。它与word2vec的更多对比请点击[GloVe vs word2vec](http://www.quora.com/How-is-GloVe-different-from-word2vec)，[GloVe & word2vec评测](http://radimrehurek.com/2014/12/making-sense-of-word2vec/)。

- 目前通过词向量可以充分发掘出"一义多词"的情况，譬如"快递"与"速递"；但对于"一词多义"，束手无策，譬如"苹果"(既可以表示苹果手机、电脑，又可以表示水果)，此时我们需要用多个词向量来表示多义词。

#### 2.3 卷积神经网络
##### 卷积
介绍卷积神经网络(convolutional neural network，简记cnn)之前，我们先看下卷积。

在一维信号中，卷积的运算，请参考[wiki](http://zh.wikipedia.org/wiki/卷积)，其中的图示很清楚。在图像处理中，对图像用一个卷积核进行卷积运算，实际上是一个滤波的过程。下面是卷积的数学表示：

$$f(x,y)*w(x,y) = \sum_{s=-a}^{a} \sum_{t=-b}^{b} w(s,t) f(x-s,y-t)$$

f(x,y)是图像上点(x,y)的灰度值，w(x,y)则是卷积核，也叫滤波器。卷积实际上是提供了一个权重模板，这个模板在图像上滑动，并将中心依次与图像中每一个像素对齐，然后对这个模板覆盖的所有像素进行加权，并将结果作为这个卷积核在图像上该点的响应。如下图所示，卷积操作可以用来对图像做边缘检测，锐化，模糊等。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/convolution1.png)
![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/convolution2.png)

图17. 卷积操作示例

##### 什么是卷积神经网络
卷积神经网络是一种特殊的、简化的深层神经网络模型，它的每个卷积层都是由多个卷积滤波器组成。它最先由lecun在LeNet[40]中提出，网络结构如下图所示。在cnn中，图像的一小部分（局部感受区域）作为层级结构的最低层的输入，信息再依次传输到不同的层，每层通过多个卷积滤波器去获得观测数据的最显著的特征。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/lenet5.png)

图18. Lenet5网络结构图


卷积神经网络中的每一个特征提取层（卷积层）都紧跟着一个用来求局部平均与二次提取的计算层（pooling层），这种特有的两次特征提取结构使网络在识别时对输入样本有较高的畸变容忍能力。如下图所示，就是一个完整的卷积过程[21]。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/convolution6.png)

图19. 一次完整的卷积过程


它的特殊性体现在两点：(1)局部感受野(receptive field)，cnn的神经元间的连接是非全连接的；(2)同一层中同一个卷积滤波器的权重是共享的（即相同的）。局部感受野和权重共享这两个特点，使cnn网络结构更类似于生物神经网络，降低了网络模型的复杂度，减少了神经网络需要训练的参数的个数。

##### 卷积神经网络的一些细节

接下来结合文献[25]，再讲讲卷积神经网络的一些注意点和问题。

- 激励函数，要选择非线性函数，譬如tang，sigmoid，rectified liner。在CNN里，relu用得比较多，原因在于：(1)简化BP计算；(2)使学习更快。(3)避免饱和问题(saturation issues)
- Pooling：其作用在于(1)对一些小的形态改变保持不变性，Invariance to small transformations；(2)拥有更大的感受域，Larger receptive fields。pooling的方式有sum or max。
- Normalization：Equalizes the features maps。它的作用有：(1)  Introduces local competition between features；(2)Also helps to scale activations at each layer better for learning；(3)Empirically, seems to help a bit (1-2%) on ImageNet
- 训练CNN：back-propagation；stochastic gradient descent；Momentum；Classification loss，cross-entropy；Gpu实现。
- 预处理：Mean removal；Whitening(ZCA)
- 增强泛化能力：Data augmentation；Weight正则化；在网络里加入噪声，包括DropOut，DropConnect，Stochastic pooling。
	- DropOut：只在全连接层使用，随机的将全连接层的某些神经元的输出置为0。
	- DropConnect：也只在全连接层使用，Random binary mask on weights
	- Stochastic Pooling：卷积层使用。Sample location from multinomial。
- 模型不work，怎么办？结合我自身的经验，learning rate初始值设置得太大，开始设置为0.01，以为很小了，但实际上0.001更合适。

##### 卷积神经网络在文本上的应用
卷积神经网络在image classify和image detect上得到诸多成功的应用，后文将再详细阐述。但除了图片外，它在文本分析上也取得一些成功的应用。

基于CNN，可以用来做文本分类，情感分析，本体分类等[36,41,84]。传统文本分类等任务，一般基于bag of words或者基于word的特征提取，此类方法一般需要领域知识和人工特征。利用CNN做，方法也类似，但一般都是基于raw text，CNN模型的输入可以是word series，可以是word vector，还可以是单纯的字符。比起传统方法，CNN不需要过多的人工特征。

- 将word series作为输入，利用CNN做文本分类。如下图所示[36]，该CNN很简单，共分四层，第一层是词向量层，doc中的每个词，都将其映射到词向量空间，假设词向量为k维，则n个词映射后，相当于生成一张n*k维的图像；第二层是卷积层，多个滤波器作用于词向量层，不同滤波器生成不同的feature map；第三层是pooling层，取每个feature map的最大值，这样操作可以处理变长文档，因为第三层输出只依赖于滤波器的个数；第四层是一个全连接的softmax层，输出是每个类目的概率。除此之外，输入层可以有两个channel，其中一个channel采用预先利用word2vec训练好的词向量，另一个channel的词向量可以通过backpropagation在训练过程中调整。这样做的结果是：在目前通用的7个分类评测任务中，有4个取得了state-of-the-art的结果，另外3个表现接近最好水平。

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/cnn_text_classify.png)

	图20.基于CNN的文本分类

	利用cnn做文本分类，还可以考虑到词的顺序。利用传统的"bag-of-words + maxent/svm"方法，是没有考虑词之间的顺序的。文献[41]中提出两种cnn模型：seq-cnn, bow-cnn，利用这两种cnn模型，均取得state-of-the-art结果。

- 将doc character作为输入，利用CNN做文本分类。文献[86]介绍了一种方法，不利用word，也不利用word vector，直接将字符系列作为模型输入，这样输入维度大大下降(相比于word)，有利于训练更复杂的卷积网络。对于中文，可以将汉字的拼音系列作为输入。

#### 2.4 文本分类
文本分类应该是最常见的文本语义分析任务了。首先它是简单的，几乎每一个接触过nlp的同学都做过文本分类，但它又是复杂的，对一个类目标签达几百个的文本分类任务，90%以上的准确率召回率依旧是一个很困难的事情。这里说的文本分类，指的是泛文本分类，包括query分类，广告分类，page分类，用户分类等，因为即使是用户分类，实际上也是对用户所属的文本标签，用户访问的文本网页做分类。

几乎所有的机器学习方法都可以用来做文本分类，常用的主要有：lr，maxent，svm等，下面介绍一下文本分类的pipeline以及注意点。

- 建立分类体系。
	- 分类相比于topic model或者聚类，一个显著的特点是：类目体系是确定的。而不像在聚类和LDA里，一个类被聚出来后，但这个类到底是描述什么的，或者这个类与另外的类是什么关系，这些是不确定的，这样会带来使用和优化上的困难。
	- 一般而言，类目体系是由人工设定的。而类目体系的建立往往需要耗费很多人工研究讨论，一方面由于知识面的限制，人工建立的类目体系可能不能覆盖所有情况；另一方面，还可能存在类目之间instance数的不平衡。比较好的方法，是基于目前已有的类目体系再做一些加工，譬如ODP，FreeBase等。
	- 还可以先用某种无监督的聚类方法，将训练文本划分到某些clusters，建立这些clusters与ODP类目体系的对应关系，然后人工review这些clusters，切分或者合并cluster，提炼name，再然后根据知识体系，建立层级的taxonomy。
	- 如果类目标签数目很多的话，我们一般会将类目标签按照一定的层次关系，建立类目树，如下图所示。那么接下来就可以利用层次分类器来做分类，先对第一层节点训练一个分类器，再对第二层训练n个分类器(n为第一层的节点个数)，依次类推。利用层次类目树，一方面单个模型更简单也更准确，另一方面可以避免类目标签之间的交叉影响，但如果上层分类有误差，误差将会向下传导。

		![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/taxonomy.png)

		图21. 层次类目体系

- 获取训练数据
	- 一般需要人工标注训练数据。人工标注，准确率高，但标注工作量大，耗费人力。
	- 为了减少标注代价，利用无标记的样本，提出了半监督学习(Semi-supervised Learning)，主要考虑如何利用少量的标注样本和大量的未标注样本进行训练和分类的问题。这里介绍两种常见的半监督算法，希望了解更多请参考文献[49]。
		-  Self-learning：两个样本集合，Labeled，Unlabeled。执行算法如下：

			- 用Labeled样本集合，生成分类策略F
			- 用F分类Unlabeled样本，计算误差
			- 选取Unlabeled中误差小的子集u，加入到Labeled集合。

			接着重复上述步骤。

			举一个例子：以前在做page分类器时，先对每一个类人工筛选一些特征词，然后根据这些特征词对亿级文本网页分类，再然后对每一个明确属于该类的网页提取更多的特征词，加入原有的特征词词表，再去做分类；中间再辅以一定的人工校验，这种方法做下来，效果还是不错的，更关键的是，如果发现那个类有badcase，可以人工根据badcase调整某个特征词的权重，简单粗暴又有效。
		-  Co-training：其主要思想是：每次循环，从Labeled数据中训练出两个不同的分类器，然后用这两个分类器对Unlabeled中数据进行分类，把可信度最高的数据加入到Labeled中，继续循环直到U中没有数据或者达到循环最大次数。
		- 协同训练，例如Tri-train算法：使用三个分类器.对于一个无标签样本，如果其中两个分类器的判别一致，则将该样本进行标记，并将其纳入另一个分类器的训练样本；如此重复迭代，直至所有训练样本都被标记或者三个分类器不再有变化。

	- 半监督学习，随着训练不断进行，自动标记的示例中的噪音会不断积累，其负作用会越来越大。所以如term weighting工作里所述，还可以从其他用户反馈环节提取训练数据，类似于推荐中的隐式反馈。
	- 我们看一个具体的例子，在文献[45]中，twitter利用了三种方法，user-level priors（发布tweet的用户属于的领域），entity-level priors（话题，类似于微博中的#***#），url-level priors（tweet中的url）。利用上面三种数据基于一定规则获取到基本的训练数据，再通过Co-Training获取更多高质量的训练数据。上述获取到的都是正例数据，还需要负例样本。按照常见的方法，从非正例样本里随机抽取作为负例的方法，效果并不是好，文中用到了Pu-learning去获取高质量的负例样本，具体请参考文献[58]。

		![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/training_data_acquisition.png)

		图22.文献[45]训练数据获取流程图

- 特征提取
	- 对于每条instance，运用多种文本分析方法提取特征。常见特征有：
		- 分词 or 字的ngram，对词的权重打分，计算词的一些领域特征，又或者计算词向量，词的topic分布。
		- 文本串的特征，譬如sentence vector，sentence topic等。
	- 提取的特征，从取值类型看，有二值特征，浮点数特征，离线值特征。
	- 特征的预处理包括：
		- 一般来说，我们希望instance各维特征的均值为0，方差为1或者某个有边界的值。如果不是，最好将该维度上的取值做一个变换。
		- 特征缺失值和异常值的处理也需要额外注意。
	-  特征选择，下面这些指标都可以用作筛选区分度高的特征。
		- Gini-index: 一个特征的Gini-index越大，特征区分度越高。
		- 信息增益(Information Gain)
		- 互信息(Mutual Information)
		- 相关系数(Correlation)
		- 假设检验(Hypothesis Testing)

- 模型训练
	- 模型选择：通常来说，常用的有监督模型已经足够了，譬如lr, svm, maxent, naive-bayes，决策树等。这些基本模型之间的效果差异不大，选择合适的即可。上一小节讲到cnn时，提到深度神经网络也可以用来做文本分类。深度神经网络相比较于传统方法，特征表示能力更强，还可以自学习特征。
	- 语言模型也可用于分类，针对不同的label，训练两个不同的语言模型p+(x\|y=+1)和p-(x\|y=-1)。对于一个testcase x，求解r= p+(x\|y=+1)/p-(x\|y=-1)*p(y=+1)/p(y=-1)，如果r>1，则x属于label(+1)，否则x属于label(-1)。
	- 模型的正则化：一般来说，L1正则化有特征筛选的作用，用得相对较多，除此外，L2正则化，ElasticNet regularization(L1和L2的组合)也很常用。
	- 对于多分类问题，可以选择one-vs-all方法，也可以选择multinomial方法。两种选择各有各的优点，主要考虑有：并行训练multiple class model更复杂；不能重新训练 a subset of topics。
	- model fine-tuning。借鉴文献[72]的思路(训练深度神经网络时，先无监督逐层训练参数，再有监督调优)，对于文本分类也可以采用类似思路，譬如可以先基于自提取的大规模训练数据训练一个分类模型，再利用少量的有标注训练数据对原模型做调优。下面这个式子是新的loss function，w是新模型参数，\\(w^0\\)是原模型参数，\\(l(w,b\|x_i,y_i)\\)是新模型的likelihood，优化目标就是最小化"新模型参数与原模型参数的差 + 新模型的最大似然函数的负数 + 正则化项"。

	$$min_{w,b} \frac{\delta}{2}||w-w^0||_2^2 - \frac{1-\delta}{n}\sum_{i=1}^nl(w,b|x_i,y_i) + \lambda(\alpha||w||_1+\frac{1-\alpha}{2}||w||_2^2)$$

	- model ensemble：也称"Multi-Model System"，ensemble是提升机器学习精度的有效手段，各种竞赛的冠军队伍的是必用手段。它的基本思想，充分利用不同模型的优势，取长补短，最后综合多个模型的结果。Ensemble可以设定一个目标函数(组合多个模型)，通过训练得到多个模型的组合参数(而不是简单的累加或者多数)。譬如在做广告分类时，可以利用maxent和决策树，分别基于广告title和描述，基于广告的landing page，基于广告图片训练6个分类模型。预测时可以通过ensemble的方法组合这6个模型的输出结果。

- 评测
	- 评测分类任务一般参考Accuracy，recall, precision，F1-measure，micro-recall/precision，macro-recall/precision等指标。

### 3 图片语义分析

#### 3.1 图片分类
图片分类是一个最基本的图片语义分析方法。

##### 基于深度学习的图片分类
传统的图片分类如下图所示，首先需要先手工提取图片特征，譬如SIFT, GIST，再经由VQ coding和Spatial pooling，最后送入传统的分类模型(例如SVM等)。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/tranditional-imageclassify.png)

图23. 传统图片分类流程图

传统方法里，人工特征提取是一个巨大的消耗性工作。而随着深度学习的进展，不再需要人工特征，通过深度学习自动提取特征成为一种可能。接下来主要讲述卷积神经网络在图片分类上的使用。

下图是一个经典的卷积神经网络模型图，由Hinton和他的学生Alex Krizhevsky在ILSVRC(Imagenet Large Scale Visual Recognition Competition) 2012中提出。
整个网络结构包括五层卷积层和三层全连接层，网络的最前端是输入图片的原始像素点，最后端是图片的分类结果。一个完整的卷积层可能包括一层convolution，一层Rectified Linear Units，一层max-pooling，一层normalization。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/convolution7.png)

图24. 卷积神经网络结构图

对于每一层网络，具体的网络参数配置如下图所示。InputLayer就是输入图片层，每个输入图片都将被缩放成227\*227大小，分rgb三个颜色维度输入。Layer1~ Layer5是卷积层，以Layer1为例，卷积滤波器的大小是11\*11，卷积步幅为4，本层共有96个卷积滤波器，本层的输出则是96个55\*55大小的图片。在Layer1，卷积滤波后，还接有ReLUs操作和max-pooling操作。Layer6~ Layer8是全连接层，相当于在五层卷积层的基础上再加上一个三层的全连接神经网络分类器。以Layer6为例，本层的神经元个数为4096个。Layer8的神经元个数为1000个，相当于训练目标的1000个图片类别。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/convolution_config.png)

图25. CNN网络参数配置图

基于Alex Krizhevsky提出的cnn模型，在13年末的时候，我们实现了用于广点通的图片分类和图片检索(可用于广告图片作弊判别)，下面是一些示例图。

图片分类示例：

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/image_classify.png)

图26. 图片分类示例图

图片检索示例：

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/image_search1.png) ![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/image_search2.png)

图27. 图片检索示例图

##### 图片分类上的最新进展

在ILSVRC 2012中，Krizhevsky基于GPU实现了上述介绍的，这个有60million参数的模型，赢得了第一名。这个工作是开创性的，它引领了接下来ILSVRC的风潮。2013年，Clarifai通过cnn模型可视化技术调整网络架构，赢得了ILSVRC。2014年，google也加入进来，它通过增加模型的层数（总共22层），让深度更深[48]，并且利用multi-scale data training，取得第一名。baidu最近通过更加"粗暴"的模型[44]，在GooLeNet的基础上，又提升了10%，top-5错误率降低至6%以下。具体结果如下图所示。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images//imagenet_result.png)

图28. ImageNet Classification Result

先简单分析一下"GoogLeNet"[48,51]所采用的方法：

- 大大增加的网络的深度，并且去掉了最顶层的全连接层：因为全连接层（Fully Connected）几乎占据了CNN大概90%的参数，但是同时又可能带来过拟合（overfitting）的效果。
- 模型比以前AlexNet的模型大大缩小，并且减轻了过拟合带来的副作用。Alex模型参数是60M，GoogLeNet只有7M。
- 对于google的模型，目前已有开源的实现，有兴趣请点击[Caffe+GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)。

再分析一下"Deep Image by baidu[44]"所采用的方法：

- Hardware/Software Co-design。baidu基于GPU，利用36个服务节点开发了一个专为深度学习运算的supercompter(名叫Minwa，敏娲)。这台supercomputer具备TB级的host memory，超强的数据交换能力，使能训练一个巨大的深层神经网络成为可能。

	而要训练如此巨大的神经网络，除了硬件强大外，还需要高效的并行计算框架。通常而言，都要从data-parallelism和model-data parallelism两方面考虑。

	- data-parallelism：训练数据被分成N份。每轮迭代里，各个GPU基于各自的训练数据计算梯度，最后累加所有梯度数据并广播到所有GPU。
	- model-data parallelism：考虑到卷积层参数较少但消耗计算量，而全连接层参数相对比较多。所以卷积层参数以local copy的形式被每个GPU所持有，而全连接层的参数则被划分到各个CPU。每轮迭代里，卷积层计算可以由各个GPU独立完成，全连接层计算需要由所有GPU配合完成，具体方法请参考[46]。

- Data augmentation。训练一个如此巨大的神经网络(100billion个参数)，如果没有充分的训练数据，模型将很大可能陷入过拟合，所以需要采用众多data augmentation方法增加训练数据，例如：剪裁，不同大小，调亮度，饱和度，对比度，偏色等(color casting, vignetting, lens distortion, rotation, flipping, cropping)。举个例子，一个彩色图片，增减某个颜色通道的intensity值，就可以生成多张图片，但这些图片和原图的类目是一致的，相当于增加了训练数据。

- Multi-scale training：训练不同输入图片尺度下(例如512\*512，256\*256)的多个模型，最后ensemble多个模型的输出结果。

#### 3.2 Image2text，Image2sentence
上面讲述的图片分类对图片语义的理解比较粗粒度，那么我们会想，是否可以将图片直接转化为一堆词语或者一段文本来描述。转化到文本后，我们积累相对深的文本处理技术就都可以被利用起来。

##### Image2text

首先介绍一种朴素的基于卷积神经网络的image to text方法。

- 首先它利用深度卷积神经网络和深度自动编码器提取图片的多层特征，并据此提取图片的visual word，建立倒排索引，产生一种有效而准确的图片搜索方法。
- 再充分利用大量的互联网资源，预先对大量种子图片做语义分析，然后利用相似图片搜索，根据相似种子图片的语义推导出新图片的语义。

其中种子图片，就是可以覆盖目前所有待研究图片的行业，但较容易分析语义的图片集。这种方法产生了更加丰富而细粒度的语义表征结果。虽说简单，但效果仍然不错，方法的关键在于种子图片。利用比较好的种子图片(例如paipai数据)，简单的方法也可以work得不错。下图是该方法的效果图。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/image_semantic.png)  ![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/image_semantic2.png)

图29. 图片语义tag标注示例图

上面的baseline方法，在训练数据优质且充分的情况下，可以取得很不错的图片tag提取效果，而且应用也非常广泛。但上面的方法非常依赖于训练数据，且不善于发现训练数据之外的世界。

另一个直观的想法，是否可以通过word embedding建立image与text的联系[26]。例如，可以先利用CNN训练一个图片分类器。每个类目label可以通过word2vec映射到一个embedding表示。对于一个新图片，先进行分类，然后对top-n类目label所对应的embedding按照权重(这里指这个类目所属的概率)相加，得到这个图片的embedding描述，然后再在word embedding空间里寻找与图片embedding最相关的words。

##### Image detection

接下来再介绍下image detection。下图是一个image detection的示例，相比于图片分类，提取到信息将更加丰富。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/image_detection.png)

图30. 图片detection示例

目前最先进的detection方法应该是Region-based CNN(简称R-CNN)[75]，是由Jeff Donahue和Ross Girshick提出的。R-CNN的具体想法是，将detection分为寻找object和识别object两个过程。在第一步寻找object，可以利用很多region detection算法，譬如selective search[76]，CPMC，objectness等，利用很多底层特征，譬如图像中的色块，图像中的边界信息。第二步识别object，就可以利用"CNN+SVM"来做分类识别。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/r-cnn.png)

图31. Image detection系统框图

- 给定一张图片，利用selective search方法[76]来产生2000个候选窗口。
- 然后利用CNN进行对每一个候选窗口提取特征(取全连接层的倒数第一层)，特征长度为4096。
- 最后用SVM分类器对这些特征进行分类（每一个目标类别一个SVM分类器），SVM的分类器的参数个数为：4096*N，其中N为目标的类别个数，所以比较容易扩展目标类别数。

这里有R-CNN的实现，请点击[rcnn code](https://github.com/rbgirshick/rcnn)

##### Image2sentence

那能否通过深度学习方法，直接根据image产生sentence呢？我们先看一组实际效果，如下图所示(copy from 文献[43])。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/image2sentence_example.png)

图32. image2sentence示例图

关于这个方向，最近一年取得了比较大的突破，工业界(Baidu[77]，Google[43]，Microsoft[80,81]等)和学术界(Stanford[35]，Borkeley[79]，UML[19]，Toronto[78]等)都发表了一系列论文。

简单归纳一下，对这个问题，主要有两种解决思路：

- Pipeline方法。这个思路相对直观一点，先学习到image中visual object对应的word(如上一节image detection所述)，再加上language model，就可以生成sentence。这种方法各个模块可以独立调试，相对来说，更灵活一点。如下图所示，这是microsoft的一个工作[81]，它分为三步：(1)利用上一节提到的思路detect words；(2)基于language model(RNN or LSTM)产生句子；(3)利用相关性模型对句子打分排序。

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/AIC.png)

	图33. "pipeline" image captioning

- End-to-end方法，即通过一个模型直接将image转换到sentence。google基于CNN+RNN开发了一个Image Caption Generator[43]。这个工作主要受到了基于RNN的机器翻译[27][42]的启发。在机器翻译中，"encoder" RNN读取源语言的句子，将其变换到一个固定长度的向量表示，然后"decoder" RNN将向量表示作为隐层初始值，产生目标语言的句子。

	那么一个直观的想法是，能否复用上面的框架，考虑到CNN在图片特征提取方面的成功应用，将encoder RNN替换成CNN，先利用CNN将图片转换到一个向量表示，再利用RNN将其转换到sentence。可以通过图片分类提前训练好CNN模型，将CNN最后一个隐藏层作为encoder RNN的输入，从而产生句子描述。如下图所示。

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/cnn_rnn.png)

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/cnn_lstm.png)

	图34. "CNN+LSTM" Image Caption Generator

	Li-Feifei团队在文献[35]也提到一种image2sentence方法，如下图所示。与google的做法类似，图片的CNN特征作为RNN的输入。

	![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/cnn-rnn.png)

	图35. "CNN+RNN"生成图片描述

	此方法有开源实现，有兴趣请参考：[neuraltalk](https://github.com/karpathy/neuraltalk)


#### 3.3 训练深度神经网络的tricks
考虑到图片语义分析的方法大部分都是基于深度学习的，Hinton的学生Ilya Sutskever写了一篇深度学习的综述文章[47]，其中提到了一些训练深度神经网络的tricks，整理如下：

- 保证训练数据的质量
- 使训练数据各维度数值的均值为0，方差为一个比较小的值
- 训练时使用minbatch，但不要设得过大，在合理有效的情况下，越小越好。
- 梯度归一化，将梯度值除于minbatch size。
- 设置一个正常的learning rate，validation无提升后，则将原learning rate除于5继续
- 模型参数随机初始化。如果是深层神经网络，不要设置过小的random weights。
- 如果是在训练RNN or LSTM，对梯度设置一个限值，不能超过15 or 5。
- 注意检查梯度计算的正确性
- 如果是训练LSTM，initialize the biases of the forget gates of the LSTMs to large values
- Data augmentation很实用。
- Dropout在训练时很有效，不过记得测试时关掉Dropout。
- Ensembling。训练多个神经网络，最后计算它们的预测值的平均值。

### 4 总结

#### 4.1 语义分析方法在实际业务中的使用
前面讲述了很多语义分析方法，接下来我们看看如何利用这些方法帮忙我们的实际业务，这里举一个例子，用户广告的语义匹配。

在广点通系统中，用户与广告的关联是通过定向条件来匹配的，譬如某些广告定向到"北京+男性"，那么当"北京+男性"的用户来到时，所有符合定向的广告就将被检索出，再按照"ecpm*quality"排序，将得分最高的展示给用户。但是凭借一些人口属性，用户与广告之间的匹配并不精确，做不到"广告就是想用户所想"，所以用户和广告的语义分析就将派上用场了，可以从这样两方面来说明：

- 特征提取。基于上面介绍的方法，提取用户和广告的语义特征。
	- 用户语义特征。可以从用户的搜索，购物，点击，阅读记录中发现用户兴趣。考虑到最终的用户描述都是文本，那么文本topic分析，文本分类，文本keyword提取，文本核心term提取都可以运用起来，分析出用户的语义属性，还可以利用矩阵分解和文本分类找到相似用户群。
	- 广告语义特征。在广点通里，广告可以从两个维度来描述，一方面是文本，包括广告title和landing page，另一方面是广告展示图片。利用文本和图片的语义分析方法，我们可以提取出广告的topic，类目，keyword，tag描述。

- 语义匹配。提取到相应的语义特征之后，怎么用于改善匹配呢？
	- 用户-广告的语义检索。基于keyword、类目以及topic，对广告建立相应的倒排索引，直接用于广告检索。
	- 用户-广告的语义特征。分别提取用户和广告的语义特征，用于计算用户-广告的relevance，pctr，pcvr，达到精确排序。

#### 4.2 Future
对于文本和图片的语义分析，可以看到：最近几年，在某些任务上，基于深度学习的方法逐渐赶上并超过了传统方法的效果。但目前为止，对于深度学习的发掘才刚刚开始，比较惊艳的神经网络方法，也只有有限几种，譬如CNN，RNN，RBM等。

上文只是介绍了我们在工作中实践过的几个小点，还有更多方法需要我们去挖掘：

- Video。Learn about 3D structure from motion。如文献[19]所示，研究将视频也转换到自然语言。
- Deep Learning + Structured Prediction，用于syntactic representation。

#### 4.3 总结

上文主要从文本、图片这两方面讲述了语义分析的一些方法，并结合个人经验做了一点总结。

原本想写得更全面一些，但写的时候才发现上面所述的只是沧海一粟，后面还有更多语义分析的内容之后再更新。另外为避免看到大篇理论就头痛，文中尽可能不出现复杂的公式和理论推导。如果有兴趣，可以进一步阅读参考文献，获得更深的理解。谢谢。

### 5 参考文献

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

