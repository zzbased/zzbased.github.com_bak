title: "Expectation-Maximization algorithm"
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# EM算法随笔

## EM overview

The EM algorithm belongs to a broader class of alternating minimization algorithms.

EM is one such hill-climbing algorithm that converges to a local maximum of the likelihood surface.

As the name suggests, the EM algorithm alternates between an expectation and a maximization step. The “E step” finds a lower bound that is equal to the log-likelihood function at the current parameter estimate θ_k. The “M step” generates the next estimate θ_k+1 as the parameter that maximizes this greatest lower bound.

![](em_image1.png)

x是隐变量。

![](em_q_function.png)

![](em_formula2.png)

参考文献：

- Zhai chengxiang老师的经典EM note：[A Note on the Expectation-Maximization (EM) Algorithm](http://www.cs.ust.hk/~qyang/Teaching/537/PPT/em-note.pdf)

- Shane M. Haas的[The Expectation-Maximization and Alternating Minimization Algorithms](http://www.mit.edu/~6.454/www_fall_2002/shaas/summary.pdf)

## EM于PLSA

PLSA的图模型：

![](plsa_graph_model.png)

PLSA的生成过程：

![](plsa_procedure.png)

(di,wj)的联合分布为：

![](plsa_formula1.png)

PLSA的最大似然函数为：

![](plsa_likelihood.png)

注意上式中，第一项的完整形式为：
\\(\sum_{i=1}^N{\sum_{j=1}^M{n(d_i,w_j) log(p(d_i))}}\\)。

对于这样的包含"隐含变量"或者"缺失数据"的概率模型参数估计问题，我们采用EM算法。这两个概念是互相联系的，当我们的模型中有"隐含变量"时，我们会认为原始数据是"不完全的数据"，因为隐含变量的值无法观察到；反过来，当我们的数据incomplete时，我们可以通过增加隐含变量来对"缺失数据"建模。

EM算法的步骤是：

- E步骤：Given当前估计的参数条件下，求隐含变量的后验概率。
an expectation (E) step where posterior probabilities are computed for the latent variables, based on the current estimates of the parameters。
- M步骤：最大化Complete data对数似然函数的期望，此时我们使用E步骤里计算的隐含变量的后验概率，得到新的参数值。
a maximization (M) step, where parameters are updated based on the so-called expected complete data log-likelihood which depends on the posterior probabilities computed in the E-step。

两步迭代进行直到收敛。

这里是通过最大化"complete data"似然函数的期望，来最大化"incomplete data"的似然函数，以便得到求似然函数最大值更为简单的计算途径。

PLSA的E-Step：

![](plsa_e_step.png)

PLSA的M-step，M-step的推导过程请参考下面的文献。

![](plsa_m_step.png)


参考文献：

- [Unsupervised Learning by Probabilistic Latent Semantic Analysis](http://www.cs.bham.ac.uk/~pxt/IDA/plsa.pdf)

- [概率语言模型及其变形系列(1)-PLSA及EM算法](http://blog.csdn.net/yangliuy/article/details/8330640)


## EM于GMM

PRML第9章。

## EM于HMM
