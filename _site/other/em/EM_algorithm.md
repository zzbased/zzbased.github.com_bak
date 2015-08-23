title: "Expectation-Maximization algorithm"
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# EM算法随笔

## EM overview

The EM algorithm belongs to a broader class of **alternating minimization algorithms**.

EM is one such hill-climbing algorithm that converges to a local maximum of the likelihood surface.

As the name suggests, the EM algorithm alternates between an expectation and a maximization step. The “E step” finds a lower bound that is equal to the log-likelihood function at the current parameter estimate θ\_k. The “M step” generates the next estimate θ\_k+1 as the parameter that maximizes this greatest lower bound.

![](em_image1.png)

x是隐变量。下面变换中用到了著名的Jensen不等式

![](em_q_function.png)

![](em_formula2.png)

参考文献：

- Zhai chengxiang老师的经典EM note：[A Note on the Expectation-Maximization (EM) Algorithm](http://www.cs.ust.hk/~qyang/Teaching/537/PPT/em-note.pdf)

- Shane M. Haas的[The Expectation-Maximization and Alternating Minimization Algorithms](http://www.mit.edu/~6.454/www_fall_2002/shaas/summary.pdf)

## EM算法细节

### Jensen不等式
Jensen不等式表述如下：

如果f是凸函数，X是随机变量，那么：E[f(X)]>=f(E[X])

特别地，如果f是严格凸函数，当且仅当X是常量时，上式取等号。

![](jensen_inequality.jpg)

Jensen不等式应用于凹函数时，不等号方向反向。

### 极大似然

给定的训练样本是{x(1),...,x(m)}，样本间独立，我们想找到每个样例隐含的类别z，能使得p(x,z)最大。似然函数为：

![](em_likelihood.png)

极大化上面似然函数，需要对函数求导。但里面有"和的对数"，求导后形式会非常复杂（自己可以想象下log(f1(x)+ f2(x)+ f3(x)+…)复合函数的求导）。所以我们要做一个变换，如下图所示，经过这个变换后，"和的对数"变成了"对数的和"，这样计算起来就简单多了。

![](jensen_transform.png)

怎么变换来的呢？其中Q(z)表示隐含变量z的某种分布。由于f(x)=log(x)为凹函数，根据Jensen不等式有：f(E[X]) >= E[f(X)]，即：

![](jensen_transform2.png)


OK，通过上面的变换，我们求得了似然函数的下届。我们可以优化这个下届，使其逼近似然函数(incomplete data)。

按照这个思路，我们要找到等式成立的条件。根据Jensen不等式，要想让等式成立，需要让随机变量变成常数值，这里得到： 

![](equality_condition.png)

![](q_condition.png)

![](p_condition.png)

![](q_p_z_relation.png)

从而，我们推导出：在固定其他参数后，Q(z)的计算公式就是z的后验概率。这一步也就是所谓的E步，求出Q函数，表示的是完全数据对数似然函数相对于隐变量的期望，而得到这个期望，也就是求出z的后验概率P(z|x，θ)。

M步呢，就是极大化Q函数，也就是优化θ的过程。

归纳下来，EM算法的基本步骤为：E步固定θ，优化Q；M步固定Q，优化θ。交替将极值推向最大。

### 为什么EM是有效的?

蓝线代表当前参数下的L函数，也就是目标函数的下界，E步的时候计算L函数，M步的时候通过重新计算θ得到L的最大值。

![](em_prove1.png)

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

统计机器学习-HMM那一章节

## 更多参考文献
- [你所不知道的EM - by erikhu](https://github.com/zzbased/zzbased.github.com/blob/master/_posts/doc/我所理解的EM算法.docx)
- [EM算法原理详解与高斯混合模型](http://blog.csdn.net/lansatiankongxxc/article/details/45646677)
- [从最大似然到EM算法浅解](http://blog.csdn.net/zouxy09/article/details/8537620)
