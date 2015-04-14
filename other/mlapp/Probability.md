title: "Probability温习"
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Probability温习

Types of machine learning：

Supervised learning：Classification，Regression

Unsupervised learning：Discovering clusters(k-means)，Discovering latent factors(PCA)，Discovering graph structure，Matrix completion，

Some basic concepts：

parametric model：the model have a fixed number of parameters。e.g. Logistic regression
non-parametric model：the number of parameters grow with the amount of training data。e.g. k-nearest neighbors。

The curse of dimensionality：

probability mass function，i.e. 概率分布函数，针对离散随机变量。
probability density function，i.e. 概率密度函数，针对连续随机变量。

![](bayes_rule.png)

![](discriminative_vs_generative.png)

概念：cumulative distribution function，probability density function，quantile(分位点)。具体请参考下图：

![](cdf_vs_pdf.png)

### discrete distributions

binomial distribution:

![](binomial_distribution.png)

multinomial distribution:

![](multinomial_distribution.png)

Poisson distribution：

![](Poisson_distribution.png)

### continuous distributions

Gaussian (normal) distribution:

![](Gaussian_distribution.png)

Student t distribution:

![](student_distribution.png)

Laplace distribution:

![](Laplace-distribution.png)

Outliers对Gaussian,Student,Laplace的影响：

![](outliers_gaussian_t_laplace.png)

gamma distribution:

![](gamma_distribution.png)
![](gamma_function.png)

Three distribution:

![](three_gamma_distribution.png)

beta distribution:

![](beta_distribution.png)



