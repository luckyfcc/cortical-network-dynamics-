# 论文方法复现代码任务清单

本文档将论文 *Cortical network dynamics and neural decoding of fine motor complexity via fNIRS and attention-based deep learning* 的方法拆解为可复现的代码工程任务。建议按以下目录组织项目：

```text
data/
preprocess/
features/
models/
experiments/
results/
```

复现目标包括：fNIRS 预处理、HbO/HbR 提取、Pearson 功能连接矩阵构建、图论指标计算、Attention-BiLSTM 分类、LOSO 跨被试验证、混淆矩阵和 ROC 曲线生成。

## data/

### 目标

负责原始数据、实验元数据、通道布局、事件标注和中间数据索引的统一管理。

### 建议目录

```text
data/
  raw/
    sub-01/
    sub-02/
    ...
  events/
    sub-01_events.csv
    ...
  metadata/
    subjects.csv
    channel_layout.csv
    task_info.csv
  processed/
    sub-01_hbo_hbr.npz
    ...
```

### 任务清单

- [ ] 建立 `subjects.csv`，记录被试编号、性别、年龄、惯用手、是否纳入分析。
- [ ] 建立 `task_info.csv`，记录五类任务标签：

| label | task | complexity |
|---:|---|---|
| 1 | fist clenching | low |
| 2 | hand flipping | low |
| 3 | paper tearing | medium |
| 4 | bottle-cap twisting | medium |
| 5 | soybean picking with chopsticks | high |

- [ ] 建立 `channel_layout.csv`，记录 24 个 fNIRS 通道的源、探测器、左右半球、脑区标签，例如 M1、PMC。
- [ ] 建立每个被试的 `events.csv`，至少包含：

```text
subject,trial,task,label,onset,duration
sub-01,1,fist_clenching,1,32.0,5.0
sub-01,1,hand_flipping,2,49.0,5.0
...
```

- [ ] 将原始 fNIRS 光强数据放入 `data/raw/`。如果原始数据来自 Homer2 支持格式，保留 `.nirs` 或原始厂商格式；如果转为 Python 处理，建议统一保存为 `.npz` 或 `.h5`。
- [ ] 编写 `data/README.md`，说明采样率为 50 Hz、通道数为 24、任务时长为 5 s、任务间休息为 10 s、每个被试 2 个 trial。

### 关键输入输出

- 输入：原始光强数据、事件标注、通道布局。
- 输出：可被预处理脚本读取的标准化数据索引。

## preprocess/

### 目标

实现 fNIRS 原始光强到 HbO/HbR 浓度变化的完整预处理流程。论文使用 MATLAB 2021 + Homer2 toolbox；Python 复现可使用 `mne-nirs`、`mne`、`scipy`、`pywt` 和自定义函数实现等价流程。

### 建议目录

```text
preprocess/
  preprocess_subject.py
  quality_control.py
  optical_density.py
  motion_correction.py
  filtering.py
  mbll.py
  cbsi.py
  block_average.py
```

### fNIRS 预处理如何实现

论文预处理流程如下，建议逐步实现并保存每一步质控图：

1. 通道质量筛选

   对应 Homer2 `EnPruneChannel`。

   - 平均光强范围：`0.01 <= mean_intensity <= 1`
   - 信噪比阈值：`SNR >= 2`
   - 输出每个通道的质量标记。

   建议实现：

   ```python
   mean_intensity = intensity.mean(axis=0)
   snr = intensity.mean(axis=0) / intensity.std(axis=0)
   good_channels = (mean_intensity >= 0.01) & (mean_intensity <= 1.0) & (snr >= 2)
   ```

2. 光强转光密度

   对应 Homer2 `hmrIntensity2OD`。

   对每个波长、每个通道计算：

   ```python
   od = -np.log(intensity / baseline_intensity)
   ```

   其中 `baseline_intensity` 可取全程均值或静息基线均值。为了接近 Homer2，建议使用每通道全程均值作为归一化参考，并在文档中固定该选择。

3. 运动伪迹检测

   对应 Homer2 `hmrMotionArtifactByChannel`。

   论文参数：

   - `tMotion = 0.5 s`
   - `tMask = 1 s`
   - `std_thresh = 5`
   - `amp_thresh = 0.05`
   - `fs = 50 Hz`

   实现要点：

   - 将 `tMotion` 转为 `25` 个采样点。
   - 将 `tMask` 转为 `50` 个采样点。
   - 在滑动窗口内检测光密度突变。
   - 若窗口内变化幅度超过 `amp_thresh` 或超过 `std_thresh * std(diff_signal)`，标记为运动伪迹。

4. 小波运动伪迹校正

   对应 Homer2 `hmrMotionCorrectWavelet`。

   论文参数：

   - 小波基：Daubechies-2，即 `db2`
   - 分解层数：4
   - 阈值系数：`iqr = 0.1`

   建议使用 `PyWavelets`：

   ```python
   coeffs = pywt.wavedec(od_channel, wavelet="db2", level=4)
   # 对 detail coefficients 做基于 IQR 的异常系数抑制
   corrected = pywt.waverec(coeffs_corrected, wavelet="db2")
   ```

5. 生理噪声滤波

   对应论文 `BandPassFilters`，滤波范围为 `0-0.1 Hz`。

   Python 中可实现为低通滤波：

   ```python
   sos = scipy.signal.butter(
       N=3,
       Wn=0.1,
       btype="lowpass",
       fs=50,
       output="sos",
   )
   od_filt = scipy.signal.sosfiltfilt(sos, od_corrected, axis=0)
   ```

   注意：严格的 `0 Hz` 高通不可实现。复现时应说明使用低通 `0.1 Hz`；如果需要去除慢漂移，可额外比较 `0.01-0.1 Hz`，但主结果应固定一种方案。

6. 光密度转 HbO/HbR 浓度

   对应 Homer2 `hmr_OD2Conc`，基于修正 Beer-Lambert 定律。

   需要输入：

   - 两个波长：760 nm、850 nm
   - 源-探距离：30 mm
   - 差分路径长度因子 DPF
   - 消光系数矩阵

   每个通道求解：

   ```text
   ΔOD(λ) = ε_HbO(λ) * ΔHbO * d * DPF(λ)
          + ε_HbR(λ) * ΔHbR * d * DPF(λ)
   ```

   转为矩阵形式：

   ```python
   delta_conc = np.linalg.pinv(E * distance * dpf) @ delta_od_two_wavelengths
   ```

   输出形状建议统一为：

   ```text
   hbo: n_times x 24
   hbr: n_times x 24
   ```

7. CBSI 校正

   对应 Homer2 `hmrMotionCorrectCbsi`，利用 HbO 和 HbR 通常负相关的关系增强信号质量。

   可实现为：

   ```python
   alpha = np.std(hbo, axis=0) / np.std(hbr, axis=0)
   hbo_cbsi = 0.5 * (hbo - alpha * hbr)
   hbr_cbsi = -(1 / alpha) * hbo_cbsi
   ```

   复现时需保存 CBSI 前后的 HbO/HbR 质控图。

8. 分段与块平均

   对应 Homer2 `HmrBlockAverage`。

   - 根据 `events.csv` 提取每个任务 onset 后 5 s 窗口。
   - 每个任务窗口长度：`5 s * 50 Hz = 250` 点。
   - 可保存 trial-level epoch，也可对同一任务做 block average。

### HbO/HbR 如何提取

- [ ] 从预处理后的浓度数据中分别保存 HbO 和 HbR。
- [ ] 推荐文件格式：

```python
np.savez(
    "data/processed/sub-01_hbo_hbr.npz",
    hbo=hbo,                 # n_times x 24
    hbr=hbr,                 # n_times x 24
    fs=50,
    channels=channel_names,
)
```

- [ ] 为机制分析提取每个任务 5 s HbO/HbR epoch：

```text
epochs_hbo: n_epochs x 250 x 24
epochs_hbr: n_epochs x 250 x 24
labels: n_epochs
subjects: n_epochs
```

- [ ] 为分类模型保留连续 HbO 时序，先做 LOSO 划分和训练集标准化，再滑动窗口切片，避免同一被试信息泄漏。

## features/

### 目标

构建任务相关 HbO 激活特征、Pearson 功能连接矩阵和图论指标。

### 建议目录

```text
features/
  activation.py
  connectivity.py
  graph_metrics.py
  export_features.py
```

### 如何构建 Pearson 功能连接矩阵

论文设置：

- 节点：24 个 fNIRS 通道。
- 边：通道间 HbO 时间序列 Pearson 相关系数。
- 每个任务取 5 s HbO 窗口，即 `250 x 24`。
- 每个样本得到 `24 x 24` 功能连接矩阵。

实现任务：

- [ ] 输入单个任务 epoch：`epoch_hbo.shape == (250, 24)`。
- [ ] 对 24 个通道两两计算 Pearson 相关。
- [ ] 将对角线设为 0 或 1，并在后续图分析前设为 0。
- [ ] 保存每个被试、trial、task 的 FC 矩阵。

示例代码：

```python
def pearson_fc(epoch_hbo: np.ndarray) -> np.ndarray:
    fc = np.corrcoef(epoch_hbo, rowvar=False)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(fc, 0.0)
    return fc
```

阈值化：

- [ ] 扫描阈值 `0.10-0.50`，步长 `0.05`。
- [ ] 主分析使用论文选择的阈值 `0.25`。
- [ ] 建议只保留正相关边：

```python
adj = (fc > threshold).astype(int)
np.fill_diagonal(adj, 0)
```

注意：如果要保留负相关边，需要提前定义解释方式；论文主要报告连接强度随复杂度增加，建议主复现使用正相关阈值。

### 如何计算 Cp、Lp、Eg、Eloc、Cd 等图论指标

建议使用 `networkx` 或 `bctpy`。为贴近脑网络分析，优先考虑 `bctpy`；若项目依赖简单，可用 `networkx` 自行实现。

#### Cp：聚类系数

定义：所有节点聚类系数的平均值。

```python
G = nx.from_numpy_array(adj)
cp = nx.average_clustering(G)
```

#### Lp：特征路径长度

定义：节点间最短路径长度的平均值。

若图不连通，建议按最大连通子图计算，并记录不连通比例：

```python
if nx.is_connected(G):
    lp = nx.average_shortest_path_length(G)
else:
    largest = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    lp = nx.average_shortest_path_length(largest)
```

#### Eg：全局效率

定义：所有节点对最短路径长度倒数的平均值。

```python
eg = nx.global_efficiency(G)
```

#### Eloc：局部效率

定义：每个节点邻域子图全局效率的平均值。

```python
eloc = nx.local_efficiency(G)
```

#### Cd：连接强度 / 连接密度

论文中 `Cd` 用于反映整体功能连接强度。复现时建议同时保存两个版本，避免术语歧义：

```python
upper = fc[np.triu_indices_from(fc, k=1)]
cd_strength = upper[upper > threshold].mean()       # 阈值后平均连接强度
cd_density = adj.sum() / (adj.shape[0] * (adj.shape[0] - 1))  # 有向写法；无向图可等价解释为边密度
```

主结果建议报告：

- `Cd_strength`：阈值以上边的平均 Pearson 强度。
- `Cd_density`：二值网络边密度。

### 输出文件

```text
features/
  outputs/
    fc_matrices.npz
    graph_metrics.csv
```

`graph_metrics.csv` 建议字段：

```text
subject,trial,task,label,threshold,Cp,Lp,Eg,Eloc,Cd_strength,Cd_density
```

## models/

### 目标

实现 Attention-BiLSTM，用多通道 HbO 时间序列识别五类精细运动任务。

### 建议目录

```text
models/
  attention_bilstm.py
  dataset.py
  train.py
  evaluate.py
  baselines.py
```

### 如何实现 Attention-BiLSTM

论文输入：

- 信号：HbO 多通道时间序列。
- 采样率：50 Hz。
- 滑动窗口长度：200 点，即 4 s。
- 滑动步长：25 点，即 0.5 s。
- 输入形状：`batch x time x channels = batch x 200 x 24`。
- 输出类别：5 类任务。

推荐 PyTorch 模型：

```python
class AttentionBiLSTM(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=64, num_layers=1, num_classes=5, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        h, _ = self.lstm(x)                    # batch x time x hidden*2
        scores = self.attn(h).squeeze(-1)      # batch x time
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(h * weights.unsqueeze(-1), dim=1)
        logits = self.fc(self.dropout(context))
        return logits, weights
```

训练设置：

- [ ] 损失函数：`CrossEntropyLoss`
- [ ] 优化器：`Adam`
- [ ] 学习率：`0.001`
- [ ] L2 正则：`weight_decay=5e-3`
- [ ] Dropout：建议从 `0.3-0.5` 调参。
- [ ] Early stopping：监控验证集 loss 或 macro-F1。
- [ ] 每个 LOSO fold 使用 3 个随机种子重复训练。

### 数据集切片任务

- [ ] 在每个 LOSO fold 中先划分训练被试和测试被试。
- [ ] `RobustScaler` 只能在训练被试连续 HbO 上拟合。
- [ ] 用训练集 scaler 变换训练和测试被试连续 HbO。
- [ ] 标准化后再做滑动窗口。
- [ ] 每个窗口标签来自其所属任务事件。
- [ ] 丢弃跨越任务边界或休息期的窗口。

窗口生成示例：

```python
window_size = 200
step_size = 25

for start in range(task_start, task_end - window_size + 1, step_size):
    stop = start + window_size
    X.append(hbo_scaled[start:stop, :])
    y.append(label - 1)
```

注意：单个任务期只有 5 s，即 250 点。用 4 s 窗口、0.5 s 步长时，每个任务 epoch 可得到 3 个窗口：

```text
0.0-4.0 s
0.5-4.5 s
1.0-5.0 s
```

## experiments/

### 目标

组织预处理、特征提取、模型训练、LOSO 验证和统计分析的可重复运行脚本。

### 建议目录

```text
experiments/
  configs/
    attention_bilstm.yaml
    graph_metrics.yaml
  run_preprocess.py
  run_features.py
  run_loso.py
  run_statistics.py
  run_all.sh
```

### 如何做 LOSO 跨被试验证

LOSO 即 Leave-One-Subject-Out。假设共有 15 名被试，每次留出 1 名被试作为测试集，其余 14 名作为训练集。

实现步骤：

1. 读取所有被试 ID。
2. 循环选择 `test_subject`。
3. `train_subjects = all_subjects - test_subject`。
4. 只用训练被试拟合 `RobustScaler`。
5. 对训练和测试连续 HbO 做相同标准化。
6. 生成训练窗口和测试窗口。
7. 从训练被试中再划分少量验证集用于 early stopping。建议按 subject 或 trial 划分，避免窗口随机泄漏。
8. 训练 Attention-BiLSTM。
9. 在留出被试上评估。
10. 每个 fold 使用 3 个随机种子重复，保存每次预测结果。

伪代码：

```python
for test_subject in subjects:
    train_subjects = [s for s in subjects if s != test_subject]

    scaler = RobustScaler()
    scaler.fit(concat_continuous_hbo(train_subjects))

    for seed in [2026, 2027, 2028]:
        set_seed(seed)
        train_dataset = build_windows(train_subjects, scaler)
        test_dataset = build_windows([test_subject], scaler)

        model = AttentionBiLSTM()
        train_with_early_stopping(model, train_dataset, val_dataset)
        metrics, predictions = evaluate(model, test_dataset)

        save_fold_result(test_subject, seed, metrics, predictions)
```

### 实验配置建议

`experiments/configs/attention_bilstm.yaml`：

```yaml
sampling_rate: 50
input_signal: hbo
num_channels: 24
num_classes: 5
window_size: 200
step_size: 25
optimizer: adam
learning_rate: 0.001
weight_decay: 0.005
batch_size: 32
max_epochs: 200
patience: 20
hidden_dim: 64
num_layers: 1
dropout: 0.5
seeds: [2026, 2027, 2028]
cv: loso
```

`experiments/configs/graph_metrics.yaml`：

```yaml
sampling_rate: 50
num_channels: 24
epoch_duration_sec: 5
thresholds: [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
main_threshold: 0.25
connectivity: pearson
signal: hbo
```

## results/

### 目标

保存可复查的模型预测、指标表格、混淆矩阵、ROC 曲线、功能连接图和图论统计结果。

### 建议目录

```text
results/
  metrics/
    loso_metrics.csv
    graph_metrics_summary.csv
  predictions/
    loso_predictions.csv
  figures/
    confusion_matrix.png
    roc_curve.png
    fc_heatmaps/
    graph_metrics_by_task.png
  logs/
    train_logs/
```

### 如何生成混淆矩阵

任务：

- [ ] 汇总所有 LOSO fold、所有随机种子的测试集预测。
- [ ] 保存字段：

```text
subject,seed,y_true,y_pred,prob_0,prob_1,prob_2,prob_3,prob_4
```

- [ ] 使用 `sklearn.metrics.confusion_matrix` 生成 5 类混淆矩阵。
- [ ] 同时生成原始计数矩阵和按真实标签归一化矩阵。

示例：

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4], normalize="true")

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_norm,
    display_labels=["Fist", "Flip", "Tear", "Cap", "Chopsticks"],
)
disp.plot(cmap="Blues", values_format=".2f")
plt.savefig("results/figures/confusion_matrix.png", dpi=300, bbox_inches="tight")
```

### 如何生成 ROC 曲线

五分类任务需要使用 one-vs-rest ROC。

任务：

- [ ] 将 `y_true` 转换为 one-hot。
- [ ] 使用模型输出概率 `prob_0` 到 `prob_4`。
- [ ] 分别计算每个类别的 ROC 和 AUC。
- [ ] 计算 micro-average 和 macro-average AUC。

示例：

```python
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

y_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
y_score = probs  # n_samples x 5

for class_idx in range(5):
    fpr, tpr, _ = roc_curve(y_bin[:, class_idx], y_score[:, class_idx])
    class_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {class_idx + 1} AUC={class_auc:.3f}")

fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, linestyle="--", label=f"micro AUC={auc_micro:.3f}")

plt.plot([0, 1], [0, 1], color="gray", linestyle=":")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("results/figures/roc_curve.png", dpi=300, bbox_inches="tight")
```

### 指标汇总

每个 LOSO fold、每个 seed 保存：

```text
test_subject,seed,accuracy,precision_macro,recall_macro,f1_macro,precision_micro,recall_micro,f1_micro,auc_macro,auc_micro
```

最终报告：

- Accuracy mean ± std
- Precision mean ± std
- Recall mean ± std
- F1-score mean ± std
- AUC mean ± std
- 95% confidence interval

95% CI 可按 fold-level 或 seed-fold-level 计算，但必须在报告中说明统计单位。建议优先以 15 个被试的 fold 平均结果作为统计单位。

## 推荐执行顺序

1. `data/`：整理原始数据、事件文件、通道布局。
2. `preprocess/`：完成单被试 HbO/HbR 提取，并输出质控图。
3. `features/`：基于 5 s HbO epoch 构建 Pearson FC 和图论指标。
4. `models/`：实现 Attention-BiLSTM 和滑动窗口数据集。
5. `experiments/`：实现 LOSO 训练评估脚本。
6. `results/`：生成指标表、混淆矩阵、ROC 曲线和网络指标图。

## 最小可复现里程碑

- [ ] 能读取 1 名被试原始 fNIRS 数据并输出 `hbo/hbr`。
- [ ] 能从 1 个任务 epoch 生成 `24 x 24` Pearson FC 矩阵。
- [ ] 能在阈值 `0.25` 下计算 `Cp/Lp/Eg/Eloc/Cd`。
- [ ] 能用全部被试跑通一次 LOSO，不追求最优准确率。
- [ ] 能保存 `loso_predictions.csv`。
- [ ] 能从预测结果生成混淆矩阵和 ROC 曲线。
- [ ] 能重复 3 个随机种子并汇总均值、标准差和 95% CI。

## 复现注意事项

- 图论指标用于神经机制解释，论文未将其作为分类模型输入。
- 分类模型主要使用 HbO，因为论文认为 HbO 对神经活动变化更敏感。
- RobustScaler 必须只在训练被试上拟合，避免跨被试数据泄漏。
- 滑动窗口切片必须在 LOSO 划分和标准化之后执行。
- 如果原始数据没有严格同步事件标注，需要先核对每个任务 onset，否则功能连接和分类标签都会偏移。
- `Cd` 的定义在不同工具箱中可能不完全一致，建议同时输出连接强度和边密度，并在结果中明确采用哪一个作为主指标。
- 若 Python 实现与 Homer2 结果存在差异，应优先检查 DPF、消光系数、滤波边界、小波阈值和 CBSI 实现。
