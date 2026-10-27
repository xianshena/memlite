# 来源与复现导航

[总索引](../../EXPERIMENTS.zh-CN.md) · [总报告](README.zh-CN.md)

下面是**源项目相对目录线索**，不是仓库下载链接。本轮仅发布文档和脱敏统计；
显微镜源码、冻结数据、原预测、权重和私人完整包尚未在这里发布，不能一键重跑显微镜。
来源文件指纹见[summary.json](summary.json)的source_fingerprints。

| 要核对什么 | 本地源项目及入口 | 已核对的事实 |
|---|---|---|
| v1的完成范围 | `architecture-memory-microscope/results/final-review-20261024/acceptance-review.json` | A36＋E15；B/C/D覆盖缺口、90项未独立验证指标 |
| BCD终态 | `architecture-memory-microscope-bcd-20261026/results/final.json`及`independent-metrics-audit.json` | 54完成，12未验证；不能当作66项能力通过 |
| B2写出失败及恢复 | `architecture-memory-microscope-b2-followon-20261026/results/final.json`与`architecture-memory-microscope-b2-recovery-20261026/results/final.json` | 12项已有结果恢复，恢复0次参数更新 |
| v1.1方案 | `architecture-memory-microscope-v11-20261026/protocol.lock.json`、`plan.json` | 三种子、1,024步、验证选模、主要比较及统计规则 |
| 独立数据规则 | 同项目`manifests/independent-data-audit.json` | A/B1/B2分母、B1末后缀对照、B2奇偶操作计数 |
| 事后补全 | `architecture-memory-microscope-v11-completion-20261027/实验补全说明.md`及`combined-results/lineage.json` | 四个B1配置的事后来源，原26项不覆盖 |
| 最终30组评分 | 同项目`combined-results/runs/<id>/result.json`和`predictions.jsonl.gz`、`reports/independent-metrics-audit.json` | 60文件SHA及30组保存输出评分复核 |
| 模型参数与汇总 | `architecture-memory-microscope-v11-report-20261027/report-stats.json` | 各模型参数、三种子均值与范围、退化基线 |

## 建议阅读代码的顺序

先读v1.1的`generate_data.py`标签规则与数据合同，再读`audit_data.py`的独立规则重放。
接着读`v11_models.py`中的状态接口和门控更新，以及`models.py`的A/B2参考结构。
然后读`v11_worker.py`的损失、选模与保存，最后读`v11_audit.py`的inspect_run和paired_bootstrap。
只查看源码不会自动训练；不要把控制、传输或调度脚本当作离线评分入口执行。

拥有本地原材料时，离线核对可只读取`result.json`与`predictions.jsonl.gz`，
按预测／目标完整标签比较，再按pair_id聚合；不需要加载Torch、权重、GPU或远端连接。
本轮在独立工作目录调用CPU评分函数，未调用会写回源报告的main入口。

## 已经公开的低成本入口

现有MemLite提供[显式对象经验代码](../../../object_memory/README.md)、
[生活环境](../../../life_choice/environment.py)、[数值关系模型](../../../life_choice/gpu_model.py)、
[既有审计脚本](../../../scripts/audit_recorded_results.py)及[复现边界](../../REPRODUCING.md)。
这些链接是旧阶段的真实公开代码，不能充当新显微镜源码。

没有发布的材料不造URL，不把私人Library包或机器绝对路径作为公共复现承诺。
