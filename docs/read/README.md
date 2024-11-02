# Let's Read
> Read papers 

## 溯源图研究现状

| name | Paper | date | content | dataset and code |
| --- | --- | --- | --- | --- |
| APT-KGL 

 | APT -KGL: An Intelligent APT Detection
System Based on Threat Knowledge and
Heterogeneous Provenance Graph Learning（**TDSC、CCF-A**） | 2022 | APT -KGL使用异构出处图（HPG）构建上下文信息。并以离线方式学习HPG中每个系统实体的语义向量表示以半自动的方式从公开的威胁知识中创建虚拟APT训练样本 | [hwwzrzr/APT-KGL: APT-KGL: An Intelligent APT Detection System Based on Threat Knowledge and Heterogeneous Provenance Graph Learning (github.com)](https://github.com/hwwzrzr/APT-KGL) |
| MEGR-APT | MEGR-APT: A Memory-Efficient APT
Hunting System Based on Attack
Representation Learning
（**TIFC、CCF-A**） | 2024 | MEGR-APT通过双重过程追踪APT：（i）在图数据库上以搜索查询的形式高效提取可疑子图，以及（ii）基于图神经网络（GNN）和我们有效的攻击表示学习（ARL）进行快速子图匹配。 | [CoDS-GCS/MEGR-APT-code: MEGR-APT: A Memory-Efficient APT Hunting System Based on Attack Representation Learning (github.com)](https://github.com/CoDS-GCS/MEGR-APT-code) |
| DGCNN for ATP | A novel approach for APT attack detection based on an advanced computing（**Scientific Reports 、SCI 2区**） | 2024 |  |  |
| **ProDE** | ProDE: Interpretable APT Detection Method Based on Encoder-decoder Architecture（**ICPADS、CCF-C**） | 2023 | 通过使用编码器-解码器架构提供可解释的结果来增强APT检测。ProDE通过比较真实图和预测图的编码表示来启动检测过程。在检测到异常时，编码器-解码器模型能够将编码解码成来源图，从而揭示解码图和真实图之间的不一致性，作为可解释的结果。 |  |
| MAGIC | [MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning](https://www.notion.so/MAGIC-12e7262428b48084a3a1cf4fab83cfba?pvs=21)（***Usenix Security* Symposium、CCF-A**）  | 2024 | MAGIC，这是一种新颖且灵活的自监督APT检测方法，能够在不同级别的监督下执行多粒度检测。MAGIC利用掩蔽图表示学习来模拟良性系统实体和行为，在来源图上执行高效的深度特征提取和结构抽象。通过异常检测方法发现异常系统行为，MAGIC能够执行系统实体级别和批量日志级别的APT检测。MAGIC专门设计了处理概念漂移的模型自适应机制，并成功应用于通用条件和检测场景。 | [FDUDSDE/MAGIC: Codes and data for USENIX Security 24 paper "MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning"](https://github.com/FDUDSDE/MAGIC) |
| **RT-APT** | [**RT-APT: A Real-time APT Anomaly Detection Method for Large-scale Provenance Graph](https://www.sciencedirect.com/science/article/pii/S1084804524002133)（**‌**Journal of Network and Computer Applications、CCF-C）** | 2024 | 本文提出了一种针对大规模数据源图的新型实时APT攻击异常检测系统，名为RT-APT。首先，使用内核日志构建数据源图，并利用WL子树内核算法聚合数据源图中节点的上下文信息。通过这种方式我们获得向量表示。其次，FlexSketch算法将流式数据源图转换为一系列特征向量。最后，在良性特征向量序列上执行K均值聚类算法，每个聚类代表不同的系统状态。这样我们就可以识别系统执行过程中的异常行为。 |  |
| 还没找到原文 | Provenance-based APT campaigns detection via masked graph representation learning（**Computers & Security、CCF-B）** |  |  |  |