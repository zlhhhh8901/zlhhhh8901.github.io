最优化是一个非常实用的知识，机器学习中很多地方都涉及到最优化知识。知识并不难，但教材编的实在差劲，并不像是为求学者讲解，更像是自顾自地炫技，学起来实在费劲。因此我决定自己编一个教材。

目录：

1. 最优化基础概念
   - 数学模型的一般形式
   - 分类
2. 最优化的数学基础
3. 具体问题
   - 无约束优化问题
   - 约束优化问题
   - 线性规划
   - 二次规划
   - 罚函数法



# 一、最优化基础概念

## 1.1 数学模型的一般形式

最优化问题是决策问题，选择一些可以执行的策略来使得目标最优。一个最优化问题包括：

1. 决策变量
2. 一个或多个目标函数
3. 一个由可行策略组成的集合，可由等式或者不等式刻画

$$
\begin{cases}\min & f(x) \\ \text { s.t. } & c_i(x)=0, i \in E=\{1,2,3 \ldots . . l\} \\ & c_i(x) \leq 0, i \in I=\{l+1, l+2, \ldots . l+m\} \\ & x \in R^n\end{cases}
$$

- **目标函数**：$f(x)$
- **约束函数**：$c_i(x)$
- **可行域**：$D=\left\{x \mid c_i(x)=0, i \in E ; c_i(x) \leq 0, i \in I, x \in R^n\right\}$
  - **无约束**：$D=R^n$，即x是自由变量
  - **有约束**：$D\subset R^n$
- **可行点**：属于D的点
- $s.t.$  =  subject to   受约束于...



## 1.2 最优化问题的分类

1. 根据可行域划分：约束与无约束
2. 根据函数性质划分：
   1. 线性规划：目标函数和约束函数都是线性的
   2. 非线性规划：存在非线性的
      - 二次规划：目标函数是二次函数（不是必须为**一元**二次），约束函数是线性的
3. 根据可行域性质划分：
   1. 连续最优化：可行域内有无数且可连续变化的点
   2. 离散最优化：可行域内点是有限个
      - 整数规划：变量均为整数
      - 混合整数规划：变量一部分为整数，一部分连续变化
4. 根据目标函数性质划分：
   - 多目标规划问题：目标函数为向量函数
   - 单目标规划问题：目标函数为数量函数
5. 根据规划问题有关信息的确定性划分： 
   - 随机规划：目标函数或约束函数具有随机性
   - 模糊规划：优化问题的变量（函数）具有模糊性
   - 确定规划：目标函数和可行域都是确定的



# 二、最优化的数学基础

## 2.1 基础知识

#### 范数

#### 二次型



## 2.2 线性代数

### 2.2.1 内积

> **内积 = 数量积 = 点积**

##### 向量角度

$$
\boldsymbol{a}\cdot \boldsymbol{b}=\left| \boldsymbol{a} \right|\left| \boldsymbol{b} \right|\cos \theta
$$

- $\theta =\widehat{\boldsymbol{a},\boldsymbol{b}}$，表示向量间的夹角
- 从几何角度理解：一个向量到另一个向量的投影

##### 线代角度

$$
\boldsymbol{a}=\left[ \begin{matrix}
	a_1&		a_2&		\cdots&		a_n\\
\end{matrix} \right] \text{，} \boldsymbol{b}=\left[ \begin{matrix}
	b_1&		b_2&		\cdots&		b_n\\
\end{matrix} \right] \text{，}\boldsymbol{a}^{\mathrm{T}}\boldsymbol{b}=\sum_{i=1}^n{a_ib_i}
$$



### 2.2.2 正定与半正定

- **正定矩阵**：给定一个大小为n×n的实对称矩阵A，若对于**任意**长度为n的非零向量 $\boldsymbol{x}$ ，有 $\boldsymbol{x}^{\mathrm{T}}\boldsymbol{Ax}>0$ 恒成立，则矩阵A是一个正定矩阵
- **半正定矩阵**：$\boldsymbol{x}^{\mathrm{T}}\boldsymbol{Ax}\geqslant 0$ 

##### 判断矩阵是否正定

**方法一**：

> 不要被正定定义的公式吓到，关键点在矩阵A

<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230503155030.png" style="zoom: 67%;" />



**方法二**：利用性质——正定矩阵的所有各阶顺序主子式都大于0
<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230517173657.png"  />

##### 几何理解

若给定任意一个正定矩阵 $A\in R^{n\times n}$ 和一个非零向量 $x\in R^n$ ，则两者相乘得到的向量 $y=Ax\in R^n$ 与向量 $x$ 的夹角恒小于90°（半正定是小于等于90°）【等价于 $\boldsymbol{x}^{\mathrm{T}}\boldsymbol{Ax}>0$ 】<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230503160141.png" style="zoom: 67%;" />

> 两向量间的夹角怎么求？——等于两向量的内积 除以 两向量的二次范数的乘积（由向量的内积公式可推出）



## 2.3 多元函数分析

##### 一元函数导数

求函数在某一个点处的变化率
$$
f'\left( x_0 \right) =\lim_{\Delta x\rightarrow 0} \frac{\Delta y}{\Delta x}=\lim_{\Delta x\rightarrow 0} \frac{f\left( x_0+\Delta x \right) -f\left( x_0 \right)}{\Delta x}
$$
<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/daoshu_change.gif" style="zoom: 80%;" />

##### 多元函数的偏导数

曲面上一点 (x0, y0) 沿x轴或y轴方向的变化率。对x求偏导数时，y固定在y0。
$$
\frac{\partial f}{\partial x}\mid_{\begin{array}{c}
	x=x_0\\
	y=y_0\\
\end{array}}^{}=f_x\left( x_0,y_0 \right) =\lim_{\Delta x\rightarrow 0} \frac{f\left( x_0+\Delta x,y_0 \right) -f\left( x_0,y_0 \right)}{\Delta x}
$$

##### 方向导数

曲面上一点 (x0, y0) 沿任意方向的变化率。假设某一方向的单位向量为 $e_l=\left( \cos \alpha ,\sin \alpha \right)$ ，α为此向量与x轴正方向的夹角，根据α的不同，此向量可以表示任意方向的单位向量。
$$
\frac{\partial f}{\partial l}\mid_{x_0,y_0}^{}=f_x\left( x_0,y_0 \right) \cos \alpha +f_y\left( x_0,y_0 \right) \sin \alpha
$$
<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230502162406.png" style="zoom:67%;" />

> 方向导数公式的证明过程暂略

上式可以看成两个向量的内积：令
<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230502163135.png" style="zoom: 67%;" />



##### 梯度

梯度是一个**向量**，表示函数在某一点处的方向导数沿梯度方向可取得最大值，即函数在该点处沿梯度的方向变化最快，变化率（梯度的模）最大

$$\nabla f\left( \boldsymbol{x} \right) =\mathbf{grad}f\left( \boldsymbol{x} \right) =\left[ \begin{array}{c}
	\frac{\partial f\left( \boldsymbol{x} \right)}{\partial x_1}\\
	\frac{\partial f\left( \boldsymbol{x} \right)}{\partial x_2}\\
	\vdots\\
	\frac{\partial f\left( \boldsymbol{x} \right)}{\partial x_n}\\
\end{array} \right]$$



##### 梯度与方向导数的关系

设f(x)具有连续的一阶偏导数，则它在点x0处沿**d**方向的一阶偏导数为：（el是d方向的单位向量） 

$$\frac{\partial f}{\partial l}\mid_{x_0,y_0}^{}=\mathbf{\nabla }f\left( x_0,y_0 \right) \cdot \boldsymbol{e}_l=\left| \mathbf{\nabla }f\left( x_0,y_0 \right) \right|\cos \theta \text{，}\theta =\left( \widehat{\mathbf{\nabla }f\left( x_0,y_0 \right) , \boldsymbol{e}_l} \right)$$



##### Hessian矩阵

> 名字很多：海森矩阵，黑塞矩阵，Hessian矩阵等

求二阶偏导，结果是个矩阵

$$\nabla^2 f(x)=\left(\begin{array}{cccc}\frac{\partial^2 f(x)}{\partial^2 x_1} & \frac{\partial^2 f(x)}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f(x)}{\partial x_1 \partial x_n} \\ \frac{\partial^2 f(x)}{\partial x_2 \partial x_1} & \frac{\partial^2 f(x)}{\partial^2 x_2} & \cdots & \frac{\partial^2 f(x)}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 f(x)}{\partial x_n \partial x_1} & \frac{\partial^2 f(x)}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f(x)}{\partial^2 x_n}\end{array}\right)$$

假设函数f(x)在**x0**处有二阶连续偏导，若

- 在**x0**处，梯度(向量)等于0<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230519143547.png" style="zoom: 67%;" />
- 且在**x0**处，海森矩阵正定

则该点为**严格局部最小点**



#### 泰勒多项式

>本文无推导过程，推导过程见：[泰勒公式（泰勒展开式）通俗+本质详解](https://blog.csdn.net/qq_38646027/article/details/88014692)
>泰勒多项式也叫泰勒展开式，泰勒公式，*近似多项式*等

##### 通俗理解

泰勒展开式是用一个函数在某点的信息，描述其附近取值的公式。如果函数足够平滑，在已知函数在某一点的各阶导数值的情况下，泰勒公式可以利用这些导数值来做系数，构建一个多项式近似函数，求得在这一点的邻域中的值。

##### 定义

> 注意区分**二元**泰勒展开式与**二阶**泰勒展开式

① **单变量函数的泰勒展开式定义**：如果函数 f(x) 在含有 x0 的某个开区间 (a,b) 内具有直到 (n+1) 阶导数，则对任意的 x 属于 (a,b) ，有
$$
f\left( x \right) =f\left( x_0 \right) +f'\left( x_0 \right) \left( x-x_0 \right) +\frac{f''\left( x_0 \right)}{2!}\left( x-x_0 \right) ^2+\cdots +\frac{f^{\left( n \right)}\left( x_0 \right)}{n!}\left( x-x_0 \right) ^n+R_n\left( x \right)
$$
其中的 Rn(x) 称为余项，即误差。

② **多变量函数的泰勒展开式定义**：H(x)是Hessian矩阵
$$
f(\mathbf{x})=f\left(\mathbf{x}_k\right)+\left[\nabla f\left(\mathbf{x}_k\right)\right]^T\left(\mathbf{x}-\mathbf{x}_k\right)+\frac{1}{2 !}\left[\mathbf{x}-\mathbf{x}_k\right]^T H\left(\mathbf{x}_k\right)\left[\mathbf{x}-\mathbf{x}_k\right]+o^n
$$



## 2.4 ✨凸函数与凸优化

#### 凸集与仿射集

- 仿射集的概念：一个集合是仿射集，当且仅当集合中经过任意两点的**直线**上的点仍在集合中，即对任意 $x,y\in S,0\leqslant \theta \leqslant 1$，有 $\theta x+\left( 1-\theta \right) y\in S$ 
- 凸集的概念：一个集合是凸集当且仅当该集合中任意两点的**连线上的所有点（即线段)** 仍然属于该集合，即对任意 $x,y\in S,\theta \in R$，有 $\theta x+\left( 1-\theta \right) y\in S$ 

> 连接$x_1$和$x_2$的线段上的任意一点$x$，都可以表示为：$x = \lambda x_1 + (1 - \lambda) x_2$，其中$\lambda \in [0,1]$
>
> 这是因为，连接$x_1$和$x_2$的线段上包含了$x_1$和$x_2$这两个端点，当$\lambda=0$时，$x=(1-\lambda)x_2 = x_2$；当$\lambda=1$时，$x= \lambda x_1 = x_1$。则当$\lambda \in (0,1)$时，$x$表示的是$x_1$与$x_2$之间的某个中间点。

- 凸集的性质： 设$C_1,C_2\subset R^n$是凸集，α属于R，则
  1. $C_1\cap C_2=\left\{ x|x\in C_1,x\in C_2 \right\}$是凸集
  2. $C_1\pm C_2=\left\{ x\pm y|x\in C_1,y\in C_2 \right\}$是凸集
- 仿射集对集合的要求包括了凸集对集合的要求，因此可以说仿射集比凸集要求更高。**仿射集一定是凸集**。



#### 凸组合

<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230423164530.png" style="zoom: 50%;" />



#### 凸包

<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230423164713.png" style="zoom:50%;" />

同理，仿射包与凸包不同的地方在 $\theta$ 的取值范围上



#### 凸函数

**定义**：设 f(x) 为定义在 n 维欧氏空间中某个凸集 S 上的函数，若对于任何实数 α(0<α<1) 以及 S 中的任意不同两点 $x^{(1)}$ 和 $x^{(2)}$，均有
$$
f\left(\alpha x^{(1)}+(1-\alpha) x^{(2)}\right) \leq \alpha f\left(x^{(1)}\right)+(1-\alpha) f\left(x^{(2)}\right)
$$

- 严格凸函数：小于等于号改为小于号
- 凹函数：不等式方向改变

**几何理解**：

<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230517162250.png" style="zoom: 80%;" />

- $f\left(x^{(1)}\right)$ 与 $f\left(x^{(2)}\right)$ 的凸组合——蓝色的线段
- $x^{(1)}$ 和 $x^{(2)}$ 的凸组合——两点间的线段
- f($x^{(1)}$ 和 $x^{(2)}$ 的凸组合)——红色的曲线

**重要性质**：*凸函数的局部最小点就是全局最小点*



#### 凸函数的判断方法

**利用一阶条件判断**

设 f(x) 在凸集 S 上有一阶连续偏导数，则 f(x) 为 S 上的凸函数的充要条件为：对于任意不同两点，均有
$$
f\left(x^{(2)}\right) \geq f\left(x^{(1)}\right)+\nabla f\left(x^{(1)}\right)^T\left(x^{(2)}-x^{(1)}\right)
$$

> 换种形式更好理解：
>
> $$\frac{f\left( x1 \right) -f\left( x2 \right)}{x1-x2}\geqslant \text{导数}$$

**利用二阶条件判断**

设 f(x) 在凸集 S 上有二阶连续偏导数，则 f(x) 为 S 上的凸函数的充要条件为：f(x) 的Hessian矩阵在 S 上处处半正定

- 严格凸函数：正定
- 凹函数：半负定

**常见的凸函数**：注意Q是对称正定矩阵

<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20Image%2020230505202616_477.png" style="zoom: 50%;" />

#### 凸优化

**定义**：可行域为凸集，目标函数为凸函数的优化问题

**具体形式**：

$$\left\{\begin{array}{l}
\min f(x) \\
h_i(x)=0, \quad i=1,2, \cdots, m \\
g_j(x) \geq 0, j=1,2 ; \cdots, \ell
\end{array}\right.$$

1. f(x)是凸函数
2. g(x)是凹函数（注意是大于等于0）
3. h(x)是线性函数

**可行域** $S=\left\{ x\in R^n|-g_j\left( x \right) \leqslant 0, h_i\left( x \right) =0 \right\}$ 为凸集

> 为什么这个可行域为凸集？
> 性质：凸函数的水平集均为凸集
> 水平集：$L_a=\left\{ x|f\left( x \right) \leqslant a,x\in C \right\}$

很多时候要化简后才能知道是否为凸优化问题<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230322183434_718.png" style="zoom: 50%;" />

**重要性质**：局部最优解就是全局最优解

**常见凸优化问题**：

- [[线性规划之单纯形法|线性规划]]
- [[约束非线性规划|凸二次规划]]：目标函数是二次型函数且约束函数是仿射函数（线性约束）

**凸优化与[[拉格朗日对偶问题与KKT和凸优化|KKT条件]]**：
KKT条件是凸优化问题最优点的充分必要条件，也就是说在凸规划中通过 KKT 条件可以找到最优解




# 三、无约束优化问题

## 3.1 最优性条件

> 为什么要介绍最优性条件？
> 原因之一：最优性条件可以指导算法的设计。例如梯度法，牛顿法，拟牛顿法等，这些算法的设计都是考虑如何能够**收敛到最优性条件**。收敛到最优性条件在很多情况下比直接去求解极值要容易的多。

**一阶必要条件**：设函数一阶连续可微，若x0是无约束问题一个局部解，则x0处梯度为0

**二阶必要条件**：设函数二阶连续可微，若x0是无约束问题一个局部解，则x0处梯度为0，且Hessian矩阵为[[正定与半正定|半正定]]

> 正定则为严格局部解

**注意判断目标函数是否为凸函数。若为凸函数，则一阶必要条件就是充分必要条件。**

> “充分条件”，其实就是指，条件能够充分的证明结论的成立
> “必要条件”，是指一个条件的成立，是结论成立必须要的一个条件
> 例如：“张三是学生”这个条件，是“张三是小学生”这个结论的必要条件



## 3.2 无约束优化问题的三大关键点

> 本文只介绍利用这三大关键点的基本思路解决无约束优化问题的方法，但其实还有其他的方法，比如函数逼近法、坐标(变量)轮换法等等

1.  **搜索方向**
2.  **搜索步长**——一维搜索方法：专门解决寻找步长的问题（假定搜索方向已知）
    1. *精确线性搜索*：寻找目标函数在特定方向上取得最小值的精确步长
    2. *非精确线性搜索*：只需要找到一个可以满足某些条件（如Armijo条件）的步长即可【通常计算效率更高，但可能会存在收敛速度较慢或不收敛的风险】
       1. Goldstein方法
       2. Armijo方法
       3. Wolfe-Powell方法
    3. *直接搜索法*：通过枚举搜索空间中的所有可能解来寻找最优解的方法。直接搜索可能会涉及到大量的计算和存储空间，因此通常只在解空间比较小的情况下使用
       1. 0.618法（黄金分割法）
       2. 均匀搜索法
3.  **终止条件**



## 3.3 搜索步长的常见解法

#### 精确线性搜索

<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230719093447.png" style="zoom:50%;" />
求解一元问题，解为步长α：
$$
\phi\left(\alpha_k\right)=\min _{\alpha \geq 0}\left\{\phi(\alpha)=f\left(x^k+\alpha d^k\right)\right\}
$$

#### 直接搜索法

> 求解精确线性搜索的一元问题时，就可以用直接搜索法！
>
> **知识点补充——确定搜索区间**：
> **单谷函数**（也叫单峰函数）
>
> <img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230426201446.png" style="zoom:50%;" />
>
> **确定搜索区间的进退算法**：
>
> <img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230426201758.png" style="zoom:50%;" />

**直接搜索法的使用前提：找到一个包含 $\phi\left(\alpha_k\right)$ 极小点的搜索区间，且函数在这个搜索区间上是单谷函数**



##### 0.618法

![](https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/0.618%E6%B3%95.png)



##### 均匀搜索法

<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230521113732.png" style="zoom:50%;" />



##### 基于导数的二分法

<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230521113908.png" style="zoom: 50%;" />

#### 非精确线性搜索



## 3.4 求解无约束优化问题的具体方法

确定搜索方向的方法（+一维搜索方法）的不同，决定了这些方法的不同：

1. 最速下降法(梯度下降法)
2. Newton法
3. 共轭梯度法

这三个方法在找搜索方向上的思路共同点：都是利用函数的性质(本身信息，如导数，偏导数，方向导数，梯度与Hessian矩阵）找函数下降方向

##### ✨梯度下降法

> 也叫最速下降法

**原理：函数在某点上的负梯度方向是函数在该点下降最快的方向**
迭代公式：$x^{\left( k+1 \right)}=x^{\left( k \right)}+\alpha ^{\left( k \right)}d^{\left( k \right)}$

- α为步长，通过线性搜索计算得
- d为下降方向，等于各点的负梯度

**算法步骤**：先任意选取一个点作为初始点<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230520105148.png" style="zoom:50%;" />

**几何图示**：<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230426211819.png" style="zoom:50%;" />
在最速下降法中相邻的两个迭代点的梯度是彼此正交的。也即在梯度的迭代过程中，相邻的搜索方向相互垂直。

因此最速下降法向极小点的逼近路径是锯齿形路线，越接近极小点，锯齿越细，前进速度越慢。这是因为梯度是函数的局部性质，从局部上看，在该点附近函数的下降最快，但从总体上看则走了许多弯路，因此函数值的下降并不快。

-   步长值取得越大，收敛速度就会越快，但是带来的可能后果就是容易越过函数的最优点，导致发散；
-   步长取太小，算法的收敛速度又会明显降低。

<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230502231640.png" style="zoom:50%;" />

**缺点与优化**：现实难以实现如此大的计算量，优化梯度下降法

1. 减小每一次训练的计算量：如随机梯度下降法
2. 优化下降路径，用更少的步数更快到达最优点：如牛顿法



##### ✨Newton法

> 目的：希望有一个算法，既能保证一定的学习步长（步长越小越精确但计算机能力没那么强），又能更好的贴近最优下降路径

**思想**：
*一维情况*

- 灰色：到达极值点的最优路径
- 橙色：一阶的近似（[[泰勒多项式]]），距离越远偏差越大
- 绿色：二阶的近似，在一定的范围内是优于橙色的

因为在a0到x范围内，二次近似多项式比一次的更接近最优路径，因此我们求这个点x，并把这个点作为下一个迭代点，如此重复迭代 <img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230502232703.png" style="zoom: 50%;" />

*多维情况*

- 灰色：最优下降路径
- 绿色：牛顿法下降路径
- 橙色：梯度下降法下降路径

<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230502233049.png" style="zoom:50%;" />

> 矩阵不能在分母，因此这里是用-1（矩阵的逆）来表示。我们可以就理解为分母

在第 k 次迭代的迭代点$x^{\left( k \right)}$邻域内，通过泰勒展开用一个二次函数去近似代替原目标函数f(x)，然后**求出该二次函数的极小点作为对原目标函数求优的下一个迭代点**，依次类推，通过多次重复迭代，使迭代点逐步逼近原目标函数的极小点。

**具体步骤**：

<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230520110733.png" style="zoom: 50%;" />

> 注意：只有Hessian矩阵是**非奇异矩阵**（行列式不为0）时这个线性方程组才是可以计算的!
>
> 奇异与非奇异的判定方法：
>
> 1. 前提：该矩阵必须为方阵
> 2. 矩阵的**行列式不为0**，则为非奇异
> 3. n×n方阵，矩阵的秩等于n，则为非奇异
> 4. 可逆矩阵与非奇异矩阵两者等价


**优点**：对于二次正定函数，迭代一次即可以得到最优解，对于非二次函数，若函数二次性较强或迭代点已经进入最优点的较小邻域，则收敛速度也很快。

**缺点**：由于迭代点的位置是按照极值条件确定的，并未沿函数值下降方向搜索，因此，对于非二次函数，有时会使函数值上升，导致计算失败（即每一步不能保证目标函数值总是下降的）。且当Hesse矩阵奇异时无法计算。

Newton法迭代结果有三种可能

1. 极小点
2. 鞍点
3. Hessian矩阵为奇异矩阵无法计算



##### 阻尼Newton法

<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230520113612.png" style="zoom:67%;" />



##### 拟Newton法

梯度法的搜索方向只需计算函数的一阶偏导数，**计算量小**，当迭代点**远离最优点时，函数值下降很快**，但当迭代点**接近最优点时收敛速度极慢**。牛顿法的搜索方向不仅需要计算一阶偏导数，而且要计算二阶偏导数及其逆阵，**计算量很大**，但牛顿法具有二次收敛性，当**迭代点接近最优点时，收敛速度很快。**

若迭代过程先用梯度法，后用牛顿法并避开牛顿法的海赛矩阵的逆矩阵的烦琐计算，则可以得到一种较好的优化方法，这就是 “拟牛顿法” 产生的基本构想。



##### 共轭梯度法

共轭梯度法是针对正定二次函数提出的一种优化方法（但也适用于一维搜索精确的一般可微函数）其性质：
（1）产生的搜索方向是下降方向；
（2）不必计算Hesse矩阵，只计算目标函数值和梯度；
（3）具有二次终止性

<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230520114426.png" style="zoom: 67%;" />

> G是正定矩阵
>
> 共轭梯度法除了计算下降方向方法不同，其迭代公式与梯度下降法类似



##### 对比各种方法

各个算法的性能对比如下： **可靠性**：牛顿法较差，因为它对目标函数要求太高，解题成功率较低。  
**有效性**：坐标变换法和梯度法的计算效率较低，因为它们从理论上不具有二次收敛性。 **简便性**：牛顿法和拟牛顿法的程序编制较复杂，牛顿法还占用较多的存储单元。

在选用无约束优化方法时，一方面要考虑优化方法的特点，另一方面要考虑目标函数的情况。 1、一般而言，对于**维数较低或者很难求得导数**的目标函数，使用**坐标轮换法**较合适。 2、对于**二次性较强的目标函数**，使用**牛顿法**效果好。 3、对于**一阶偏导数易求**的目标函数，使用**梯度法**可使程序编制简单，但精度不宜过高 4、综合而言，共轭梯度法和 DFP 法具有较好的性能。



# 四、约束优化问题

1. KKT条件：最优点的一阶必要条件
2. 罚函数法：**构造惩罚函数将约束问题转化为无约束问题**进行求解，根据**惩罚函数的不同**分为
   1. 外点法
   2. 内点法
   3. 乘子法：在罚函数的基础上增加了拉格朗日乘子项，从而称为増广拉格朗日函数。这里只讨论等式约束的情况
3. 二次规划（QP问题）
   1. **KKT法：是约束非线性规划的一般方法**，但只能用于二次规划的**等式**约束
   2. 有效集法（也叫积极集法）：二次规划专属的可解决不等式约束（可包含等式约束）的解决方法

### 罚函数之外点法

**只有等式约束**时，可将该问题转化为如下无约束问题形式：
$$
P\left( x,\sigma \right) =f\left( x \right) +\sigma\sum_i^m{\left[ h_i\left( x \right) \right] ^2}
$$

- $\sigma$：罚因子
- P()：惩罚函数
- $\sigma\sum_i^m{\left[ h_i\left( x \right) \right] ^2}$：惩罚项

**只有小于等于的不等式约束**时，可转化为如下无约束问题形式：
$$
P\left( x,\sigma \right) =f\left( x \right) +\sigma\sum_i^m{\left[ \max \left( 0,h_i\left( x \right) \right) \right] ^2}
$$
**都有时**，将上面的等式约束和不等式约束中的罚项加在一起即可构造惩罚函数：
$$
P\left( x,\sigma \right) =f\left( x \right) +\sigma\left( \sum_i^m{\left[ h_i\left( x \right) \right] ^2}+\sum_j^l{\left[ \max \left( 0,g_j\left( x \right) \right) \right] ^2} \right) 
$$

例：<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230521194517.png" style="zoom:50%;" />



### 罚函数之内点法

内点法**只考虑不等式约束**问题

<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230522152239.png" style="zoom:50%;" />

内点法就是通过**在严格内点集合中进行迭代**得到最优解，这也是内点法这一说法的来历，需要注意的是**内点法得到的最优解并不一定是全局最优解，因为内点法只是在严格内点中迭代，而全局最优解有可能落在边界上**。但是对于最优解不落在边界的问题，内点法能够得到最优解，并且对于那些最优解落在边界上的问题，内点法也能够获得较好的近似解。

障碍函数的作用是惩罚靠近可行域边界的 x 点，即那些使 g(x)=0 的点，当 x 靠近这些边界的时候，障碍项会变得很大，从而使得其不满足障碍函数最小。

<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230521201046.png" style="zoom: 67%;" />



### 罚函数之乘子法

乘子罚函数是在罚函数的基础上增加了拉格朗日乘子项，从而称为増广拉格朗日函数

- **只有等式约束**：$$
  \varphi(x, \lambda, \sigma)=f(x)+\sum_{i=1}^m \lambda_i h_i(x)+\frac{\sigma}{2} \sum_{i=1}^m\left[h_i(x)\right]^2
  $$
- **只有不等式约束**：<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230522154121.png" style="zoom:67%;" />
- **都有**：<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230522154153.png" style="zoom:67%;" />



### 二次规划问题

**二次规划定义**：目标函数为二次函数，约束条件为线性约束<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230505204407.png" style="zoom:50%;" />

- **矩阵Q为[[正定与半正定|半正定]]矩阵，则为凸二次规划问题**
- 正定，严格凸二次规划问题

**二次规划问题（QP问题）的解法**：

1. 等式约束二次规划问题

2. KKT法：约束非线性规划的一般方法

       <img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230522175445.png" style="zoom:50%;" />

   2. <img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230522175853.png" style="zoom:50%;" />

3. 变量消除法（高中方法）

4. **不等式约束二次规划问题**

- 凸二次规划的有效集法（也叫积极集法）



#### 有效集法

**基本思想**：

1. <img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230517201048.png" style="zoom: 67%;" />
   - 黑色箭头指向约束的区域，蓝色五角星是是全局最优点
   - 左图，最优点在不等式范围之内，但有没有这个不等式约束都可以求出来，所以被称为**无效约束**
   - 右图，不等式约束范围求出来的最优解在绿色点上，而这个绿色最优点在不等式约束的等式约束上，所以**有效的不等式约束是等式约束**

> 极值点在可行域外时，最优解出现在边界条件上

2. 所以我们的思路是：把这些真正起作用的约束找出来，将不等号换成等号，其他的无效不等式直接抛弃掉——**将不等式约束二次规划问题变为等式约束二次规划问题**
3. 思想确定了，那下一步的关键问题就是——如何寻找有效集（也就是有效的不等式约束）呢？

**寻找有效集**：[从55:40开始看](https://www.bilibili.com/video/BV1vQ4y1P77A?p=2&vd_source=9f8d13d25bf9216916258d388abf3f5c)
<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230522195243.png" style="zoom: 50%;" />
<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/Pasted%20image%2020230522195235.png" style="zoom:50%;" />
<img src="https://cdn.jsdelivr.net/gh/zlhhhh8901/hello-world@main/img/1.png" style="zoom: 33%;" />

[举例：有效集法_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1vQ4y1P77A?p=3&vd_source=9f8d13d25bf9216916258d388abf3f5c)