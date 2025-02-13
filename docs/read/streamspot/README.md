# StreamSpot

> Fast Memory Anomaly Detection in Streaming Heterogeneous Graph
> 

假设我们有以下节点类型和边类型：

- **节点类型**：`Process` (进程), `File` (文件), `Socket` (套接字)
- **边类型**：`read`, `write`, `fork`, `connect`

我们有一个流式图，边的到达顺序如下（每条边表示为 `<源节点, 源类型, 目标节点, 目标类型, 时间戳, 边类型, 流标签>`）：

1. `<P1, Process, F1, File, t1, read, G1>`
2. `<P1, Process, S1, Socket, t2, connect, G1>`
3. `<P2, Process, F2, File, t3, write, G2>`
4. `<P1, Process, P2, Process, t4, fork, G1>`
5. `<P2, Process, S1, Socket, t5, connect, G2>`

假设我们有两个图 `G1` 和 `G2`，它们的边交错到达。

### 算法步骤

### 1. Shingling（生成 shingle）

假设我们选择 `k=1`，即每个 shingle 是通过 1-hop 广度优先遍历生成的字符串。我们为每个节点生成 shingle：

- 对于节点 `P1`，shingle 为 `Process-read-File` 和 `Process-connect-Socket`。
- 对于节点 `P2`，shingle 为 `Process-write-File` 和 `Process-connect-Socket`。

### 2. 构建 Shingle 向量

每个图的 shingle 向量记录了图中每个 shingle 的出现次数。假设我们有以下 shingle 向量：

- `G1` 的 shingle 向量：
    - `Process-read-File`: 1
    - `Process-connect-Socket`: 1
    - `Process-fork-Process`: 1
- `G2` 的 shingle 向量：
    - `Process-write-File`: 1
    - `Process-connect-Socket`: 1

### 3. Sketching（生成 sketch）

为了节省内存，我们使用 **StreamHash** 将 shingle 向量映射到固定大小的 sketch 向量。假设我们选择 sketch 的大小为 `L=4`，并使用 4 个哈希函数 `h1, h2, h3, h4`，每个哈希函数将 shingle 映射到 {+1, -1}。

假设哈希函数的映射结果如下：

- `h1(Process-read-File) = +1`, `h2(Process-read-File) = -1`, `h3(Process-read-File) = +1`, `h4(Process-read-File) = -1`
- `h1(Process-connect-Socket) = -1`, `h2(Process-connect-Socket) = +1`, `h3(Process-connect-Socket) = -1`, `h4(Process-connect-Socket) = +1`
- `h1(Process-fork-Process) = +1`, `h2(Process-fork-Process) = +1`, `h3(Process-fork-Process) = -1`, `h4(Process-fork-Process) = -1`
- `h1(Process-write-File) = -1`, `h2(Process-write-File) = -1`, `h3(Process-write-File) = +1`, `h4(Process-write-File) = +1`

根据这些哈希值，我们可以计算每个图的 sketch 向量：

- `G1` 的 sketch 向量：
    - `h1`: +1 (Process-read-File) + -1 (Process-connect-Socket) + +1 (Process-fork-Process) = +1 → `sign(+1) = +1`
    - `h2`: -1 + +1 + +1 = +1 → `sign(+1) = +1`
    - `h3`: +1 + -1 + -1 = -1 → `sign(-1) = -1`
    - `h4`: -1 + +1 + -1 = -1 → `sign(-1) = -1`
    - 因此，`G1` 的 sketch 向量为 `[+1, +1, -1, -1]`
- `G2` 的 sketch 向量：
    - `h1`: -1 (Process-write-File) + -1 (Process-connect-Socket) = -2 → `sign(-2) = -1`
    - `h2`: -1 + +1 = 0 → `sign(0) = +1`（假设 sign(0) = +1）
    - `h3`: +1 + -1 = 0 → `sign(0) = +1`
    - `h4`: +1 + +1 = +2 → `sign(+2) = +1`
    - 因此，`G2` 的 sketch 向量为 `[-1, +1, +1, +1]`

### 4. 动态维护与聚类

随着新边的到达，StreamSpot 会动态更新图的 sketch 向量。假设我们有一条新边 `<P2, Process, S1, Socket, t5, connect, G2>`，它会影响 `G2` 的 shingle 向量。我们更新 `G2` 的 sketch 向量：

- 新 shingle：`Process-connect-Socket`（已经存在，计数增加）
- 更新 `G2` 的 sketch 向量：
    - `h1`: -1 (Process-write-File) + -1 (Process-connect-Socket) = -2 → `sign(-2) = -1`
    - `h2`: -1 + +1 = 0 → `sign(0) = +1`
    - `h3`: +1 + -1 = 0 → `sign(0) = +1`
    - `h4`: +1 + +1 = +2 → `sign(+2) = +1`
    - `G2` 的 sketch 向量保持不变：`[-1, +1, +1, +1]`

### 5. 异常检测

StreamSpot 使用聚类模型来检测异常图。假设我们有两个聚类中心：

- 聚类中心 1：`[+1, +1, -1, -1]`（代表正常行为）
- 聚类中心 2：`[-1, +1, +1, +1]`（代表另一种正常行为）

当新图到达时，StreamSpot 计算其 sketch 向量与每个聚类中心的距离。如果距离超过某个阈值，则将其标记为异常。

### 总结

通过这个例子，我们可以看到 StreamSpot 的工作流程：

1. **Shingling**：生成 shingle 并构建 shingle 向量。
2. **Sketching**：使用 StreamHash 将 shingle 向量映射到固定大小的 sketch 向量。
3. **动态维护**：随着新边的到达，增量更新 sketch 向量。
4. **聚类与异常检测**：通过聚类模型检测异常图。

StreamSpot 能够在流式场景下高效地处理异构图，并实时检测异常。