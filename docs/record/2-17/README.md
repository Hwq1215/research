# 2025.5.30-2025.6.6

# 科研

## APT检测存在的问题：

1. **假阳率过高（**良性样本和恶意样本之间的嵌入区分度不足，良性样本和恶意样本各自的嵌入分布不集中，共同特征没被充分挖掘**）**
2. **鲁棒性不足（存在对抗的逃逸和中毒攻击能逃避检测）**
3. **恶意标签的数据集不足**

## 目前大方向：

1. 给无监督学习加入恶意标签数据，实现**半监督（弱监督）的学习方法**，加大良性样本和恶意样本之间的嵌入区分度
2. 通过少量的标签数据，先基于规则组装数据，后续尝试使用深度学习算法，**构建攻击图生成模型**
3. 结合联邦学习

## 第一个方向

- [x]  引入恶意子图，添加对比损失增加良性节点和恶意节点的嵌入差异
- [x]  紧凑损失，使得良性节点和恶意节点的嵌入相近 (采用三元组的损失)
- [ ]  分配权重，加大困难样本（如相似的恶意节点和良性节点）的学习程度

## 以往工作

实现了了三个节点数据集上和两个图级数据集的实验，验证了第一个方向的第1点的成功结果，相对于MAGIC假阳率降低显著

选取了恶意节点10%的数据作为输入

### 实验

**MAGIC（无监督，baseline）**

![image.png](image.png)

![image.png](image%201.png)

**Slot（半监督，同样10%恶意数据）**

![image.png](image%202.png)

**Our Work**

在图级数据集`streamspot`，`unicorn wget`上实验

引入紧凑损失，做实验，目前设想引入对比学习常用的的三元组损失**`Triplet loss`**

![image.png](image%203.png)

$$
L = \max \left( {d\left( {a,p}\right)  - d\left( {a,n}\right)  + \text{ margin },0}\right)
$$

这里**取间隔 (margin) 为 0。**

损失 max(0, sim_an - sim_ap) 意味着只有当**锚点-负样本的相似度 (sim_an) 严格大于锚点-正样本的相似度 (sim_ap)**时，模型才会受到惩罚（产生损失）。因为良性活动或恶意行为并不意味着他们的特征是完全紧凑的或方向是完全一致的，严格的使得两者产生margin可能对模型的泛化性能造成影响，这是我们通过实验佐证的。

我们保证其他模块不变，观察修改为三元组损失后，在几个节点数据上的实验结果。

相对于以前只有单一的对比损失的优化结果，这里控制每个数据集召回率尽量一致

```yaml

theia:
	TN: 319082 -> 319109
	FN: 1
	TP: 22787
	FP: 366  -> 339

cadets:
	TN: 343270 -> 343295
	FN: 37  ->  40
	TP: 11525 -> 11522
	FP: 1057 ->  1032
	
trace:
	TN: 615980 -> 615996
	FN: 22
	TP: 61256
	FP: 41 -> 25

```

与之前做的对比损失相比，三元组损失对实验是有帮助的

### 两个权重

1. **第一重聚焦：**通过批处理困难样本挖掘**，自动、隐式地聚焦于那些在几何上**最接近恶意节点的“困难”良性样本。这直接实现了“加强与恶意节点相近的良性节点学习”的需求，且无需计算额外的权重。
2. **第二重聚焦**：通过**GMAE-VIB 混合编码器，**提供的`μ`嵌入用于稳定的重建，`σ`用于鲁棒的采样，而计算得到的每一个节点的`KL散度`，即每个节点的“`编码代价`”，能**对那些在**信息论上更复杂、更罕见的“困难”良性样本，进行一次显式的、额外的加权。

重点在第二重聚焦，用自监督的方式评估良性样本的内在困难度，然后用这个困难度指导有监督的良恶性边界学习。

# 本周工作

1. 尝试将原有的GAE变成GVAE，第一步是尝试将原来的嵌入和隐藏层再通过一个**VIB头**，加入**信息瓶颈正则项，**来获得更符合真实分布的嵌入空间，目前还在做实验和调整网络和参数。
    
    ```python
       def forward(self, x, edge_index):
    # 第一层 GAT 层，使用 ELU 激活函数
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
    # 第二层 GAT 层
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index).squeeze()
    # 生成均值和方差
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            return mu, logvar
    ```
    
2. 完成了论文introuction章节的内容，总结梳理了有关研究背景、文献内容,准备下周的BackGround的写作
    
    ![image.png](image%204.png)
    

## 计划

1. 完成目前的设想，并做实验
2. 开始编写论文的Realted Work
3. 寻找新数据做实验，Flash有一个`Drapa Optc`的`groundtruth`，应该可以作为补充数据集