# Headline

> An awesome project.

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

![[Pasted image 20230503155030.png|500]]

**方法二**：利用性质——正定矩阵的所有各阶顺序主子式都大于0（如何把二次函数转化为线性代数形式？——[[线性代数之二次型]]）
![[Pasted image 20230517173657.png|550]]


##### 几何理解

若给定任意一个正定矩阵 $A\in R^{n\times n}$ 和一个非零向量 $x\in R^n$ ，则两者相乘得到的向量 $y=Ax\in R^n$ 与向量 $x$ 的夹角恒小于90°（半正定是小于等于90°）【等价于 $\boldsymbol{x}^{\mathrm{T}}\boldsymbol{Ax}>0$ 】![[Pasted image 20230503160141.png|600]]

> 两向量间的夹角怎么求？——等于两向量的内积 除以 两向量的二次范数的乘积（由向量的内积公式可推出）



## 2.3 多元函数分析

##### 一元函数导数

求函数在某一个点处的变化率
$$
f'\left( x_0 \right) =\lim_{\Delta x\rightarrow 0} \frac{\Delta y}{\Delta x}=\lim_{\Delta x\rightarrow 0} \frac{f\left( x_0+\Delta x \right) -f\left( x_0 \right)}{\Delta x}
$$
![[daoshu_change.gif]]

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
![[Pasted image 20230502162406.png|500]]

> 方向导数公式的证明过程暂略

上式可以看成两个向量的内积：令
![[Pasted image 20230502163135.png|600]]

##### 梯度

梯度是一个**向量**，表示函数在某一点处的方向导数沿梯度方向可取得最大值，即函数在该点处沿梯度的方向变化最快，变化率（梯度的模）最大
$$
\nabla f\left( \boldsymbol{x} \right) =\mathbf{grad}f\left( \boldsymbol{x} \right) =\left[ \begin{array}{c}
	\frac{\partial f\left( \boldsymbol{x} \right)}{\partial x_1}\\
	\frac{\partial f\left( \boldsymbol{x} \right)}{\partial x_2}\\
	\vdots\\
	\frac{\partial f\left( \boldsymbol{x} \right)}{\partial x_n}\\
\end{array} \right] 
$$

##### 梯度与方向导数的关系

设f(x)具有连续的一阶偏导数，则它在点x0处沿**d**方向的一阶偏导数为：（el是d方向的单位向量） 

$$
\frac{\partial f}{\partial l}\mid_{x_0,y_0}^{}=\mathbf{\nabla }f\left( x_0,y_0 \right) \cdot \boldsymbol{e}_l=\left| \mathbf{\nabla }f\left( x_0,y_0 \right) \right|\cos \theta \text{，}\theta =\left( \widehat{\mathbf{\nabla }f\left( x_0,y_0 \right) , \boldsymbol{e}_l} \right) 
$$

##### Hessian矩阵

> 名字很多：海森矩阵，黑塞矩阵，Hessian矩阵等

求二阶偏导，结果是个矩阵

$$\nabla^2 f(x)=\left(\begin{array}{cccc}\frac{\partial^2 f(x)}{\partial^2 x_1} & \frac{\partial^2 f(x)}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f(x)}{\partial x_1 \partial x_n} \\ \frac{\partial^2 f(x)}{\partial x_2 \partial x_1} & \frac{\partial^2 f(x)}{\partial^2 x_2} & \cdots & \frac{\partial^2 f(x)}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 f(x)}{\partial x_n \partial x_1} & \frac{\partial^2 f(x)}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f(x)}{\partial^2 x_n}\end{array}\right)$$

假设函数f(x)在**x0**处有二阶连续偏导，若

- 在**x0**处，梯度(向量)等于0![[Pasted image 20230519143547.png|400]]
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



## 2.4 凸函数与凸优化

#### 凸集与仿射集

- 仿射集的概念：一个集合是仿射集，当且仅当集合中经过任意两点的**直线** 上的点仍在集合中
  ![[Pasted image 20230423164204.png|500]]
- 凸集的概念：一个集合是凸集当且仅当该集合中任意两点的**连线上的所有点（即线段)** 仍然属于该集合

> 连接$x_1$和$x_2$的线段上的任意一点$x$，都可以表示为：$x = \lambda x_1 + (1 - \lambda) x_2$，其中$\lambda \in [0,1]$
>
> 这是因为，连接$x_1$和$x_2$的线段上包含了$x_1$和$x_2$这两个端点，当$\lambda=0$时，$x=(1-\lambda)x_2 = x_2$；当$\lambda=1$时，$x= \lambda x_1 = x_1$。则当$\lambda \in (0,1)$时，$x$表示的是$x_1$与$x_2$之间的某个中间点。

- 凸集的性质： ![[Pasted image 20230517181922.png]]
- 仿射集对集合的要求包括了凸集对集合的要求，因此可以说仿射集比凸集要求更高。**仿射集一定是凸集**。

#### 凸组合

![[Pasted image 20230423164530.png]]

#### 凸包

![[Pasted image 20230423164713.png]]

同理，仿射包与凸包不同的地方在 $\theta$ 的取值范围上


### 超平面

- 超平面是平面中的直线、空间中的平面的推广，是纯粹的数学概念，不是现实的物理概念
- 超平面是仿射集也是凸集

#### 从数学公式角度理解

- 二维空间的直线：$Ax+By+C=0$
- 三维空间的平面：$Ax+By+Cz+D=0$
- 以此类推——n维空间的超平面：$w_1x_1+w_2x_2+w_3x_3+...+w_nx_n+b=0$

最终定义形式：$w^{\tau}x+b=0$
**w** 和 **x** 都是 n 维**列向量**。**x** 为平面上的点；**w 为平面上的法向量**，决定了超平面的方向；b 是一个实数，代表超平面到原点的距离

#### 从向量内积角度理解

定义：$w^{\tau}x=b$

![[Pasted image 20230426194843.png]]


#### 点到平面的距离

空间内任意一点 **x**（向量）到超平面的距离：![[Pasted image 20230424153931.png]]

> 二维空间：$$\frac{\left| ax_0+by_0+c \right|}{\sqrt{a^2+b^2}}$$

### 半空间

- 超平面可以将它所在的空间分为两半，半空间在法向量的反方向
- 半空间是凸集，但不是仿射集

![[Pasted image 20230426194959.png]]


#### 凸函数

**定义**：设 f(x) 为定义在 n 维欧氏空间中某个凸集 S 上的函数，若对于任何实数 α(0<α<1) 以及 S 中的任意不同两点 $x^{(1)}$ 和 $x^{(2)}$，均有
$$
f\left(\alpha x^{(1)}+(1-\alpha) x^{(2)}\right) \leq \alpha f\left(x^{(1)}\right)+(1-\alpha) f\left(x^{(2)}\right)
$$

- 严格凸函数：小于等于号改为小于号
- 凹函数：不等式方向改变

**几何理解**：![[Pasted image 20230517162250.png]]

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
> $\frac{f\left( x1 \right) -f\left( x2 \right)}{x1-x2}\geqslant \text{导数}$

**利用二阶条件判断**

设 f(x) 在凸集 S 上有二阶连续偏导数，则 f(x) 为 S 上的凸函数的充要条件为：f(x) 的Hessian矩阵在 S 上处处半正定

- 严格凸函数：正定
- 凹函数：半负定

**常见的凸函数**：注意Q是对称正定矩阵![[Pasted Image 20230505202616_477.png|500]]

#### 凸优化

**定义**：可行域为凸集，目标函数为凸函数的优化问题

**具体形式**：![[Pasted image 20230517175126.png]]
**可行域** $S=\left\{ x\in R^n|-g_j\left( x \right) \leqslant 0, h_i\left( x \right) =0 \right\}$ 为凸集

> 为什么这个可行域为凸集？
> 性质：凸函数的水平集均为凸集
> 水平集：$L_a=\left\{ x|f\left( x \right) \leqslant a,x\in C \right\}$

很多时候要化简后才能知道是否为凸优化问题 ![[Pasted Image 20230322183434_718.png|400]]
**重要性质**：局部最优解就是全局最优解

**常见凸优化问题**：

- [[线性规划之单纯形法|线性规划]]
- [[约束非线性规划|凸二次规划]]：目标函数是二次型函数且约束函数是仿射函数（线性约束）

**凸优化与[[拉格朗日对偶问题与KKT和凸优化|KKT条件]]**：
KKT条件是凸优化问题最优点的充分必要条件，也就是说在凸规划中通过 KKT 条件可以找到最优解