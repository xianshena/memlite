![MemLite](assets/header.svg)

# MemLite：可复核的记忆条件决策实验

[English](README.md) · [结果与限制](docs/RESULTS.md) · [复现说明](docs/REPRODUCING.md)

这是一个学习驱动的早期项目：在合成场景中，观察显式经验记忆怎样改变选择。
仓库提供零参数 CPU 示例、308M 模型三种子训练的既有记录、消融以及逐决策审计工具。
目前没有已核验的外部使用或独立复现证据。

## 先运行 CPU 示例

Python 3.11+，在仓库根目录执行，前三项无需第三方依赖：

```sh
python -m examples.cpu_quickstart
python -m unittest discover -s tests -p test_object_memory.py -v
python -m scripts.audit_recorded_results
python -m pip install -r requirements-cpu.txt
python -m unittest discover -s tests -p test_life_environment.py -v
python -m scripts.audit_recorded_results --with-frozen-data
```

CPU 示例为显式状态表，没有可训练参数；它不代表重新训练了 308M 模型。

## 308M 既有结果

模型实际参数为 308,487,175。三个独立种子各训练 12 轮、948 次更新；每个种子使用
10,000 个独立训练场景，重复呈现 12 轮。三种子共用同一组 2,000 测试场景，不能称为
6,000 个独立测试场景。测试分桶为日常 1,600、经验 300、未知效用平局 100。

| Recorded result | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Daily correct / 1,600 | 1,506 | 1,505 | 1,506 |
| Daily accuracy | 94.125% | 94.0625% | 94.125% |
| Both sides correct + flipped / 200 history pairs | 200 | 200 | 200 |
| Clear-memory forced choice correct / 200 | 102 | 100 | 100 |
| Separate unknown-preference controls / 2,000 | 994 | 958 | 1,002 |

日常准确率的分母为 **daily subset n=1,600 per seed**，并非全部 2,000 个测试场景。

历史交换按 200 对计分，两边都正确且选择反转才算成功；清空记忆只计每对的原始一侧。
全测试含平局的效用正确率为 95.30%、95.25%、95.30%，不能解释为隐藏偏好预测准确率。
保留了 6,000 条行动审计记录，但完整的 200 对历史预测逐条日志未保留。

## 诚实边界

- 记忆由按主体筛选的显式前端提供，未证明自主学习写入或遗忘。
- 导航由确定性 BFS 和抽象执行完成，未证明真实视觉、机器人运动控制或生物机制。
- 候选重排对照没有独立预设数值通过阈值；保留实际偏差与失败案例。
- G0/G1a 为 CPU 诊断，后续接受门槛尚未验证。
- 4B 教师到 1.85B 学生仅探索附录：恢复阶段载入 272 个训练样本、64 个验证样本，
  教师一致率 42/64，无独立测试，不能套用 308M 成绩。
- 打包检查通过 36 项 CPU 测试，6 项 GPU 测试因缺 Torch/CUDA 跳过；未重新进行 GPU 训练。
  历史 CUDA 环境不等于完整可移植依赖锁。

源码与文档采用 [MIT](LICENSE)，合成数据和数值记录的范围见 [DATA_LICENSE](DATA_LICENSE.md)。
第三方依赖保留其原许可，详见 [声明](THIRD_PARTY_NOTICES.md)。不包含大权重、私人聊天、
申报个人资料、凭据、公司代码或角色插图。
