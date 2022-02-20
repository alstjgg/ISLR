# 2.2 Assessing Model Accuracy

---

The most importat concepts in selecting a statistical learning procedure for a specific data set.

# 2.2.1 Measuring the Quality of Fit

Evaluating the **performance** of a statistical learning method on a given data set

= Measurement of **how well predictions match observations**

## MSE (Mean Squared Error)

MSE is the most commonly used measure for regression.

$$
MSE=\frac1n\underset{i=1}{\overset{n}\Sigma}(y_i-\hat f(x_i))^2
$$

where

- $\hat f(x_i)$: prediction of the $i$th ovbservation
- $MSE \downarrow \leftrightarrow = Performance Quality \uparrow$

## Train vs Test

- $Training MSE$
    
    → computed using the **training data**, data used to fit the model
    
    → $\hat f(x_i) \sim y_i$
    
- $TestMSE$
    
    → computed using **test data**, data previously unseen
    
    → $Ave(y_0-\hat f(x_0))^2$
    
    → $(x_0, y_0)$: unseen and unused data
    

We are more interested in **test MSE** in measuring the quality of the method. However, if no test observations are available(*no test data*), then one might select a method that minimizes the **train MSE**.

Unfortunately, low training MSE does not guarantee low test MSE, as shown in the graph below.

![Untitled](2%202%20Assess%20ca9d6/Untitled.png)

- The **test MSE** initially declines as the level of flexibility increases
- At some point, the **test MSE** levels off and starts to increase
- **Training MSE** monotonously decreases
- **Test MSE** shows a *U-shape*

We must **find the flexibilty level with the minimal test MSE**. This can be done through various methods, such as *cross-validation*.

## Overfitting

When a given method yields **a small training MSE but a large test MSE**, we are said to be **overfitting** the data. In this case, a less flexible model would have yielded a smaller test MSE.

# 2.2.2 The Bias-Variance Trade-Off

## 3 Fundamental Quantities of MSE

MSE can be decomposed into the sum of 3 values.

- **variance** of $\hat f(x_0)$
    - the amount by which $\hat f$ would change if we used a different training data set
    - high variance → small changes in the training data can result in large changes in $\hat f$
    - $ModelFlexibility\propto Variance$
- squared **bias** of $\hat f(x_0)$
    - the error that is introducted by approximating a real-life problem by a much simpler model
    - $ModelFlexibility\propto 1/Bias$
- **variance of the error** terms $\epsilon$

In other words,

$$
E(y_0-\hat f(x_0))^2=Var(\hat f(x_0))+[Bias(\hat f(x))]^2+Var(\epsilon)
$$

where

- $E(y_0-\hat f(x_0))^2$: expected test MSE at $x_0$
- $Var(\epsilon)$: irreducible error

As $Var(\epsilon)$ is irreducible, we wish to **minimize $Var(\hat f(x_0))$ and $Bias(\hat f(x))$**. However, there exists a **trade-off** between these competing properties, resulting in the *U-shape* in test MSE curves.

## Trade-Off

As model flexibilty increases, variance increases and bias decreases.

The relative rate of changes of these 2 quantities determines whether the test MSE increases or decreases.

![Untitled](2%202%20Assess%20ca9d6/Untitled%201.png)

As we increase flexibility,

- Bias decreases and variance increases
- At first, bias tends to decrease faster than variance increases
    
    → test MSE declines
    
- At some point, flexibility has little impact on bias, but variance increases
    
    → test MSE increases
    

# 2.2.3 The Classification Setting

In the classification setting, we estimate $f$ on the basis of training observations ${(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)}$ where $**y_1, y_2, ..., y_n$ are qualitative**. We measure **the accuracy of $f$** by computing the proportion of mistakes that are made when applying $f$. In other words, we compute **the fraction of incorrect classifications** with

$$
\frac1n \underset{i=1}{\overset n{\Sigma}}I(y_i\neq\hat y_i)
$$

where

- $\hat y_i$: the **predicted class label** for the $i$th observation using $\hat f$
- $I(y_i\neq\hat y_i)$: **indicator variable**
    - 1 if $y_i\neq\hat y_i$, 0 if $y_i=\hat y_i$

## The Bayes Classifier

To minimie error rate, we should **assign each observation to the mot likely class given its predictor values**. In other words, assign each observation with predictor vector $x_0$ to the class $j$ where

$$
Pr(Y=j|X=x_0)
$$

is largest.

The Bayes Classifier produces **the lowest possible test error rate**, called the **Bayes error rate,** given by 

$$
1-E(\underset{j}maxPr(Y=j|X)).
$$

Usually we do not know the coniditional distribution of $Y$ given $X$, so **computing the Bayes Classifier is impossible**. Thus computing the Bayes Classifier is **an unattainable gold standard against which to compare other methods**. Many methods try to estimates the conditional distribution of $Y$ given $X$ and **classify an observation to the class with highest estimated probability**.

## K-Nearest Neighbors

Given an integer $K(K\leq 0)$ and a test observation $x_0$, the KNN Classifier 

1. **Identify $N_0$** : the $K$ points in the trainind data that are closest to $x_0$
2. Estimate the **conditional probability for class $j$** 
    
    $$
    Pr(Y=k|X=X_0)=\frac1K\underset{i\in N_0}\Sigma I(y_i=j).
    $$
    
3. Classyfy $x_0$ to the class with the largest probability from 2.

The **choice of** $K$*(choosing the level of flexibility)* is crucial to the performance of KNN classifiers.

- When $K=1$, the decision boundary is overly flexible
    
    ![Untitled](2%202%20Assess%20ca9d6/Untitled%202.png)
    
    → low bias, high variance
    
- As $K$ grows, it becomes less flexible
    
    ![Untitled](2%202%20Assess%20ca9d6/Untitled%203.png)
    
    ->low variance, high bias