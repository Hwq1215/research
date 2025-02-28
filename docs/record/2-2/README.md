# 2025.2.22-2025.2.28

# 大模型

- 完成了Unsloth的测试任务
- 提供了测试数据，完成了测试报告

# 科研

> 发现攻击图数据生成还是有些复杂，有使用黑盒查询攻击来做的，但是多为节点分类任务，按照需要添加的边的数量级来说，比较难做到，也和现有的对抗攻击有不同，也有可能思路错了，我想以MAGIC论文为基础，做一个能对抗（至少能比以往的自编码器模型要好）这种攻击的检测系统，尝试放入不同的自编码器看效果
> 
- [x]  完成了梦坤学长论文的插图
- [x]  阅读`Adversarial Attack`的相关论文
- [ ]  之后尝试找到一个对抗`Adversarial Attack`和`Evasion Attack`攻击的图自编码器，即具有鲁棒性可以最大化防御这些攻击（至少超越之前的入侵检测系统）

## Adversarial Attack on Graph Structured Data

> 只阅读了一部分，周末将其读完
> 

### 写在前面

尽管最近在图像和文本等其他领域的对抗性攻击和防御方面取得了进展，但很少有人关注涉及图结构的领域，作者在本文中，重点介绍了一组图神经网络 （GNN） 模型的图对抗攻击。这些是一系列**监督模型**（Dai et al.， 2016）在许多**转导任务**（Kipf & Welling，2016）和**归纳任务**（Hamilton et al.，2017）中取得了最先进的结果。通过**节点分类和图分类问题**的实验，我们将证明此类模型确实存在对抗样本。GNN 模型可能很容易受到此类攻击。但是有效地攻击图结构不简单，因为图是离散的，此外，图形结构的组合性质使其比文本困难得多。

### 文章的做法

文章提出了一个

1. 基于强化学习的攻击方法，该方法仅根据目标分类器的预测反馈来学习修改图结构。修改是通过在图形中**按顺序添加或删除边缘**来完成的。
2. 分层方法还用于分解二次动作空间，以使训练可行。

### Adversarial Attack（对抗攻击）的分类

1. **白盒攻击（**white box attack ，WBA**）：**在这种情况下，攻击者可以访问目标分类器的任何信息，包括预测、梯度信息等。
2. **灰盒攻击（**practical black box attack，PBA**）：**在这种情况下，只有目标分类器的预测可用。当预测置信度可访问时，我们表示此设置作为 PBA-C;如果只允许离散预测标签，我们将设置表示为 PBA-D。
3. **黑盒攻击（**restrict black box attack，RBA**）：**在这种情况下，我们只能对**某些样本**进行黑盒查询，并要求攻击者创建对其他样本的对抗性修改。

关于攻击者可以从目标分类器获得的信息量，我们可以将上述设置排序为: **WBA>PBA-C>PBA-D>RBA**

### 定义了GNN介绍了两个监督学习任务

1. 归纳（Inductive）图分类(粒度为图，测试集和训练集分离)：例子是根据药物分子图的功能对药物分子图进行分类
    
    优化目标：
    
    $$
    \mathcal{L^{(ind)}} = \frac{1}{N}\sum^{N}_{i=1}L(f^{(ind)}(G_i),y_i)
    $$
    
2. 传导的节点分类（粒度为节点，训练集和测试集在同一个图上，只是训练阶段测试集没有标签）：示例包括在 Citeseer 等引文数据库中对论文进行分类，或在 Facebook 等社交网络中对实体进行分类。
    
    优化目标：
    
    $$
    \mathcal{L}^{(tra)}=\frac{1}{N}\sum_{i=1}^{N}L(f^{(tra)}(c_i;G_0),y_i)
    $$
    

### 定义图对抗性攻击问题

给定一个学习到的分类器f和数据集(G,c,y) ∈ D，图攻击者g(·,·):g×D->g 将图G=（V，E）转化为*G̃* = (*Ṽ*, *Ẽ*)

对抗攻击目标：

$$
\begin{aligned}
 & \max_{\tilde{G}} & & \mathbb{I}(f(\tilde{G},c)\neq y) \\
 & s.t. & & \tilde{G}=g(f,(G,c,y)) \\
 & & & \mathcal{I}(G,\tilde{G},c)=1.
\end{aligned}
$$

这里ℐ(⋅, ⋅, ⋅) : 𝒢 × 𝒢 × *V* ↦ {0, 1}是一个等价指示符，它表示两个图 G 和 G ̃ 在分类语义下是否等效。

对于*G̃*有两个约束目标：

1. *G̃*是通过某种函数*g*从原图*G*、分类器*f*、节点*c*和真实标签*y*生成的
2. ℐ(*G*, *G̃*, *c*) = 1 ，即在分类语义下，原图*G*和生成的图*G̃*是等价的。
    
    假设我们知道G的语义信息，有一个黄金标准的分类器，有
    
    ℐ(*G*, *G̃*, *c*) = 𝕀(*f**(*G*, *c*) = *f**(*G̃*, *c*))
    
    如果没有语义信息，只能是最小的修改来说明
    
    $$
    \begin{gathered}
     \mathcal{I}(G,\tilde{G},c)=\mathbb{I}(|(E-\tilde{E})\cup(\tilde{E}-E)|<m) \\
     \cdot\mathbb{I}(\tilde{E}\subseteq\mathcal{N}(G,b))).
     \end{gathered}
    $$
    

注意到，修改边比修改节点要困难，因为选择一个节点只需要 O(|V |) 的复杂度，而选择一条边需要 O(|V |2)

## 修改APT检测模型的自编码器

1. 在 `GMAEModel` 的 `__init__` 方法中初始化教师网络 `t_encoder`，并设置 EMA 更新器。
2. 修改掩码生成函数，接受教师网络的输出作为输入，并基于此生成动态掩码。
3. 在训练过程中，定期更新教师网络的参数，使用 EMA 平滑策略。
4. 在损失函数中引入对比学习损失，利用教师网络的输出作为辅助信息。

> 根据ProtoMGAE: Prototype-Aware Masked Graph Auto-Encoder for Graph Representation Learning**原型感知自编码器**，对MAGIC: detecting advanced persistent threats via masked graph representation learning的网络进行了修改，初步能跑通，目前还在实验中
> 

```python
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)
```

# 国网

## 基于电力大数据的用户侧数据异常检测方法研究_李厚恩

> 孤立森林算法虽然不进行距离、密度等指标计算，大大加快了运算速度，但其无法检测局部用电异常，因此本文提出了一种将孤立森林算法和局部离群因子算法相结合的电力大数据异常检测方法。
> 

## 孤立森林

## 局部离群因子（LOF）

局部离群因子（LOF）算法是有代表性的基于密度的离群点检测方法，其主要思想是为每个数据点分配离群因子 $LOF_k(o)$，并通过离群因子的大小确定样本数据是否异常。算法步骤如下。

步骤１输入数据集Ｄ，指定ｋ值和尚未遍历的点ｏ。

步骤２根据式（２）计算点ｏ到点ｐ的第ｋ可达距离

$$
\text { reach } \_d_k(o, p)=\max \left\{d_k(o), d_k(o, p)\right\}
$$

式中，/  $\text { reach } \_d_k(o, p)$为点ｏ到点ｐ的第ｋ可达距离，$d(o, p)$为点ｏ和点ｐ之间的距离， $d_k(o)$ 为点ｏ的第ｋ距离。

步骤３根据式（３）计算点ｏ的局部可达密度$p_k(o)$

$$
p_k(o)=\frac{1}{\sum_{p \in N_k(o)} \text { reach } d_k(o, p) / N_k(o)}
$$

式中， $p_k(o)$为平均可达距离的倒数（第ｋ邻域内的点到点ｏ）。如果该点ｏ位于簇中，则该点ｏ位于点ｐ的ｋ距离邻域中的概率较大。

步骤４根据式（４）计算点ｏ的局部离群因子 $LOF_k(o)$

$$
\operatorname{LOF}_k(o)=\frac{\sum_{p \in N_k(o)}\left(\frac{\rho_k(p)}{\rho_k(o)}\right)}{N_k(o)}
$$

将计算出的 $LOF_k(o)$ 值按降序排列，前ｎ个放置的数据样本点即为异常点。

## 方案

基于孤立森林和局部离群因子算法的电力数据异常检测思想是利用主成分分析对某一类电力用户的用电数据特征集进行降维，通过孤立森林过滤掉用电数据中的全部异常后的剩余数据作为局部离群因子算法的输入，以获得更精确的异常点［１３］。本文用电数据异常检测方法的主要步骤如下。
**步骤１**利用模糊Ｃ－均值对电力用户用电数据进行特征分类，并采用主成分分析法进行降维。
**步骤２**初始化孤立森林算法参数，通过孤立森林算法进行全局异常值检测。
**步骤３**通过孤立森林过滤掉用电数据中的全局异常后，使用局部异常因子算法进行二次异常检测，识别局部异常并获得用户离群度排序。图２为异常检测方法的流程图。