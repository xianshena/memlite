# 实验文档时间线

[仓库首页](../README.zh-CN.md) · [实验总索引](EXPERIMENTS.zh-CN.md) · [按阶段](#by-stage) · [按文档提交](#by-document) · [文档目录](#catalog)

想看最近实验，先读[结构与记忆显微镜报告](experiments/microscope-20261027/README.zh-CN.md)；
想理解已公开的记忆选择训练，先读[既有结果与限制](RESULTS.md)；
想自己核对记录，先读[复现说明](REPRODUCING.md)。

本页整理于2026年10月28日，索引现有文档，不移动原文件。日期分三种：
**实验／阶段日期**来自材料明示；**报告或资料标注日**不一定是实验日；
**Git提交时间**来自提交记录，也不等于已核实的上传完成或首次公开时间。
没有证据的日期保留“未明确”，不从文件夹后缀或后续汇总日倒推。

<a id="by-stage"></a>
## 按实验阶段查找

| 日期线索 | 阶段 | 去哪里读、怎样理解日期 |
|---|---|---|
| 2026-10-21：协议日期 | 308M数值生活选择 | [冻结协议](../protocols/life_choice_original.json)明确标注该日及Asia/Shanghai；这不是全部训练起止时刻的核验。[结果与分母](RESULTS.md)、[复现说明](REPRODUCING.md)随2026-10-22提交进入本仓库 |
| 2026-10-23：阶段材料所述执行日 | 结构与记忆显微镜v1 | A36次、E15次所选运行；不代表全部原协议完成。见[分阶段记录](experiments/microscope-20261027/stages.md)的v1部分。该阶段的本仓库文档提交在10月27日 |
| 2026-10-26：补跑材料所述日期 | BCD独立补跑 | 66个计划种子槽中54完成、12未验证跳过；见[分阶段记录](experiments/microscope-20261027/stages.md)。不能把计划槽全算成完成实验 |
| 精确执行日未明确 | B2 follow-on与recovery | 12项曾在结果写出阶段失败，后来恢复已有结果；恢复期间0次优化器更新。见[分阶段记录](experiments/microscope-20261027/stages.md)与[来源导航](experiments/microscope-20261027/sources.md)。不靠源目录后缀补日期 |
| 原批次精确执行日未明确 | 显微镜v1.1原批次 | 保留26/30完成与4项超时。见[最新报告](experiments/microscope-20261027/README.zh-CN.md)；“资料截至10月27日”不能代替原批次起止日 |
| 2026-10-27：补全材料所述日期 | v1.1四个B1事后续训 | 原26项加四项事后续训，合并30/30；不能追溯写成测试前一次性30/30。见[阶段与四项来源](experiments/microscope-20261027/stages.md)、[方法限制](experiments/microscope-20261027/methods-and-limits.md) |
| 实验日未逐项明确 | G0/G1a显式对象经验、4B教师／1.85B扩参 | 已有[结果汇总](RESULTS.md)与[CPU示例说明](REPRODUCING.md)，本仓库可核实文档提交为10月22日；不要把提交日当实验日 |
| 更早阶段，日期未逐项核实 | Qwen部署、推理优化、量化、RAG、个人LoRA、原文字记忆、小狗指令理解 | 见[实验总索引](EXPERIMENTS.zh-CN.md)中的历史摘要。摘要不是全部源码、完整报告或原始记录；未公开部分不造下载链接 |

<a id="by-document"></a>
## 按报告／资料标注和文档提交查找

以下Git时间均为UTC。表中的日期不是服务器上传完成回执，也不保证当时仓库已处于公开状态。
首次提交指这些文档在当前可见Git历史中的首次记录，不是对文件在其他地方更早存在的否定。

| 材料 | 报告／资料自身的日期标注 | 可核实Git记录 |
|---|---|---|
| MemLite结果、复现、权属及首批源码说明 | 这些文档未统一单列报告生成日；生活选择协议有2026-10-21日期 | [18bcd382](https://github.com/xianshena/memlite/commit/18bcd3826ecba46077b00e4441b2b9040431a1b9)，2026-10-22 07:33:14 UTC |
| 首批文件字节／校验修正 | 文件完整性维护，不代表新增实验 | [4e8a9fb3](https://github.com/xianshena/memlite/commit/4e8a9fb3a979bf62c5b012c38aef900aa2a1426d)，2026-10-22 07:35:19 UTC |
| 中文实验总索引与显微镜分阶段报告／统计 | 总索引写明资料截至2026-10-27，统计JSON updated为2026-10-27；这不是每阶段的实验发生日 | [78770fb8](https://github.com/xianshena/memlite/commit/78770fb8248f5096a21c0603839a3228cba269e8)，2026-10-27 06:34:43 UTC |
| 本时间线与首页导航 | 整理标注2026-10-28，仅新增导航 | [本页提交历史](https://github.com/xianshena/memlite/commits/main/docs/TIMELINE.zh-CN.md)给出实际提交时间 |

<a id="catalog"></a>
## 文档目录：按用途找

### 阅读入口

- [中文首页](../README.zh-CN.md)：CPU入门、308M三种子摘要与边界。
- [English README](../README.md)：英文入口与同一批公开内容。
- [中文实验总索引](EXPERIMENTS.zh-CN.md)：将旧部署、RAG、训练和最新显微镜串起来。

### MemLite记忆与选择

- [结果、控制与失败](RESULTS.md)：G0/G1a、308M生活选择及教师扩参附录；按各自分母读成绩。
- [复现边界与命令](REPRODUCING.md)：区分低成本CPU审计、历史CUDA参考及未复现项。
- [对象经验包入口](../object_memory/README.md)、[生活选择包入口](../life_choice/README.md)：简短包说明，配合前两篇阅读。
- [生活选择原协议](../protocols/life_choice_original.json)：日期、数据划分、损失与控制的机器可读定义。

### 最新结构与记忆显微镜

- [分实验总报告](experiments/microscope-20261027/README.zh-CN.md)：A/B1/B2问题、模型、结果、失败与重复输入。
- [阶段记录](experiments/microscope-20261027/stages.md)：v1、BCD、B2恢复、v1.1原批次与四项事后续训。
- [方法与统计边界](experiments/microscope-20261027/methods-and-limits.md)：成对分母、重复测量、bootstrap及不能推出的结论。
- [来源与复现导航](experiments/microscope-20261027/sources.md)：哪些代码已公开，哪些仅有本地源项目线索。
- [统计CSV](experiments/microscope-20261027/summary.csv)：适合表格查看的模型结果。
- [统计JSON](experiments/microscope-20261027/summary.json)：分母、重复输入、比较与来源文件指纹。

### 许可与贡献

- [代码许可证](../LICENSE)、[数据范围许可](../DATA_LICENSE.md)。
- [权属与来源说明](RIGHTS_AND_LICENSES.md)、[第三方声明](../THIRD_PARTY_NOTICES.md)。
- [贡献方式](../CONTRIBUTING.md)。

<a id="coverage"></a>
## 当前目录的覆盖范围

本轮整理前仓库有15份Markdown，未包含Word或PDF文件。此前学习手册、私人完整归档或未发布原始材料，
不能从这些公开文档推断为已经上传；本页不为它们编造公开下载链接，也不补传它们。
其他访问范围的资料不并入此公开目录。原文件、结果和历史提交均保留，本轮没有训练或修改模型代码。
