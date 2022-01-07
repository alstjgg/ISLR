ISL_ch2
==================
## Statistical learning

### 2.1 definition

**Statistical learning** 은 함수 f를 찾아내기 위한 접근법의 집합

---
     
#### 2.1.1 Why Estimate f?     

      
**예측(prediction)** & **추론(inference)** 이 목적.

**예측** - 독립변수인 X를 입력값으로 받아서 실제값 Y와 최대한 근접한 예측값 Y를 산출하기 위한 방법.

**추론** - 여러 개의 혹은 하나의 독립변수 X(predictor)와 예측값 Y(response) 사이의 관계를 알아내기 위한 방법.

<br>

실제값 Y를 예측하기 위한 예측값 Y(^y)의 정확도는 **reducible error**와 **irreducible error**라는 두 개의 값에 의존한다.

**reducible error**는 통계 기법들을 활용하여 예측값 Y의 정확도를 개선함으로써 제거할 수 있는 오류.

**irreducible error**는 제거 불가능한 오류.

아무리 예측값 Y를 실제값 Y와 유사하게 만들 수 있다 하더라도, Y 자체가 독립변수인 X를 이용하여 예측할 수 없는 epsilon 의 함수이기 때문

**E(Y - \hat{Y})^2 = E[f_x - \hat{f_x}]^2 + Var(ϵ)**

**irreducible error**가 0보다 큰 이유?
1. f를 추정하기 위한 과정에서 고려하지 않은 변수를 epsilon이 (그러나 계측에 유용한) 포함하고 있음
2. epsilon의 variation(분산)이 측정불가능함.

---

#### 2.1.2 How Do We Estimate f?

실제함수 f와 최대한 근접한 값을 갖는 예측함수 f(^f)를 추정하기 위한 방법으로 **parametric method** 와 **Non-parametric method** 가 있다.

**parametric method**
1. 우선 f의 기능적인 형태와 모양에 대한 가정을 한다. 즉, 적절한 model을 선택.
2. model을 선택한 후, 학습 데이터를 활용하여 해당 model을 학습시킨다.

**Non-parametric method** 는 함수 f의 기능적인 형태에 대한 특정한 가정을 하지 않는다. 

---

#### 2.1.3 The Trade-Off Between Prediction Accuracy and Model Interpretability

일반적으로 모델의 **유연성(flexibility)** 이 커질수록, 모델의 variance가 증가하며, bias가 감소한다.    
따라서, **자유도(=유연성)** 가 큰 모델일수록 **예측**에 유리하며,    
linear model처럼 **자유도가 낮은 (즉 restrictive한)** 모델일 수록 독립변수와 종속변수 간의 관계를 추정하기 위한 **추론**에 용이하다.

---

#### 2.1.4 Supervised Versus Unsupervised Learning

독립변수(x)와 그와 관련된 종속변수(y)가 모두 있다면 **지도학습(supervised learning)** 이 가능

BUT, 독립변수(x)에 대응되는 종속변수(y)가 없는 데이터의 분석에 대해서는 **지도학습**이 불가능하기 때문에 **비지도학습(unsupervised learning)** 이 사용됨

---

#### 2.1.5 Regression Versus Classification Problems

변수들은 **정량적 변수(quantitative variables)** 또는 **범주화 변수(qualitative variables)** 로 정의된다. 

종속변수(Y)가 **정량적 변수(quantitative variables)** 인지 **범주화 변수(qualitative variables)** 인지에 따라서    
**회귀(regression)** 또는 **분류(classification)** 를 사용한다. 

* 독립변수(X)가 정량적 변수인지 범주화 변수인지는 비교적 중요하지 않다.    
  분석에 들어가기 전에 one-hot encoding과 같은 방식으로 범주화 변수의 성격을 갖는 독립변수를 정량적 변수로 변환할 수 있기 때문. 

---

### 2.2 Assessing Model Accuracy

#### 2.2.1 Measuring the Quality of Fit

**회귀**를 사용할 때, 모델의 성능을 평가하기 위해 가장 보편적으로 사용되는 지표는 **MSE(Mean Squared Error)** 이다. 

**MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{f}_xi)^2**

* 데이터 분석에서 모델의 성능을 향상시키기 위해서 줄이고자 하는 MSE는 test MSE이다.   
즉, 기존에 학습을 위해 사용된 학습 데이터에 대한 training MSE보다는 학습에 쓰이지 않은 새로운 데이터 (가령 test dataset)를 모델에 입력인자로 사용하여 정확한 예측을 하기 위함이 목적이다. 

---

#### 2.2.2 The Bias-Variance Trade-Off

보통, 모델의 자유도(유연성)가 증가할수록, 학습 데이터에 대한 MSE는 작아지지만, 검증 데이터에 대한 MSE는 U자 모양을 가진다.    

* 즉, 자유도가 일정 수준까지 커질수록 variance의 증가폭보다 bias의 감소폭이 크기 때문에 모델의 MSE가 감소하다가    
  일정 수준을 넘어가면 bias의 감소폭보다 variance의 증가폭이 커지기 때문에 모델의 MSE가 다시 증가하는 형태를 보인다. 

---

#### 2.2.3 The Classification Setting 

종속변수가 정량적 변수가 아닌 범주화 변수 일 때, 예측값 f(^f)에 대한 정확도를 정량화하기 위한 가장 일반적인 접근법은 **training error rate** 을 활용한 방식이다. 

* **training error rate** - 학습 데이터의 관측값(실제값)에 예측값 f(^f)을 적용시켰을 경우의 오차비율(?)

---

#### The Bayes Classifier

**베이즈 분류(Bayes Classifier)** 는 조건부 확률을 활용한 방식으로, 사전에 정해진 임계값을 기준으로 어떤 독립변수(x0)가    
특정 라벨(j)에 속할 조건부 확률이 임계값보다 높은지 낮은지에 따라서 분류하는 모델이다. 

* (조건부확률) Pr( Y=j | X=x0 )

---

#### K-Nearest Neighbors

**K-Nearest Neighbors** 는 검증 데이터의 독립변수(x0)와 특정 정수값을 갖는 K가 주어졌을 때, x0에 가장 가까운 K개의 학습 데이터를 찾고, x0에 가장 가까운 K개의 학습 데이터의 class 중에서 가장 많은 class로 x0를 분류하는 방식이다. 


* K가 커질수록 모델의 자유도(유연성)이 감소하며, 이에 따라 variance는 감소하지만 bias가 높아진다. 
* K가 작아질수록 모델의 자유도(유연성이) 증가하여 variance가 증가하고 bias가 감소한다. 
* 따라서, K가 클수록 모델이 분류한 결과의 경계는 선형성을 갖고, K가 작아질수록 모델이 분류한 결과의 경계는 비선형적이 된다.  



























