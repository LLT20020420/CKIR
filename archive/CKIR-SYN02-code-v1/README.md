# CKIR SYN-02 固定源码补充 Code S1

版本 CKIR-SYN02-code-v1，2026-10-24。此包用于定位论文保留模型实际使用的损失、初始化、核求解与三阶段训练代码。

## 入口

先阅读 SOURCE_INDEX.md 和 retained_config.json。运行 `python inspect_package.py` 可校验包内文件哈希、语法与函数入口，不会训练、评分或写入实验结果。

`portable/` 提供可独立导入的 CKIRText、LPHead 与两项上游损失声明，函数体与对应冻结原文一致。它们使用原有 PyTorch/CUDA 类型与运算顺序。`solvers/` 提供原样 core.py/core_v04.py，可将该目录加入 Python 模块搜索路径后导入 BoxWorkspace。

`historical/` 为当次实际执行的完整源文件快照。源文件保持原样以便核对 SHA-256，包含当时的目录约定；它们依赖当次特征、核权重、标签与检查点资产。此源码包没有将这些历史驱动重写为新的跨机器一键实验系统。

## 三阶段输入和检查点约定

第一阶段 fit 的支持数组 s 为296×1024，查询 q 为592×1024，o为296个观测类别索引。wt为296×296的支持核系数，wq为296×592的查询核系数。template/cupl/negative文字缓冲为1024×37；只训练两组37×1024正文字残差。第0项保存初始化，保留第25轮残差。

第二阶段载入第25轮文字，W/b从零开始、gamma100；使用16/1/0.03/2.51的固定几何，保留第7步。第三阶段同时载入这两组起点，重建优化器状态，更新文字与W/b7步；保留第7步的res_template、res_cupl、weight、bias、gamma和text_features。两阶段所有支持残差均由完整296支持计算，包含自身核项。第一阶段的辅助损失不用于第三阶段。

数据划分、类别顺序、每份支持ID、三份观测标签和几何权重应与原记录对应。大体积图像、特征、权重和旧预测不包含在此源码包中。论文中的已有结果没有因整理本包而重跑。

## 函数调用与依赖

实际训练入口为 historical/text_stage.py:fit、historical/aligned_head_stage.py:fit 和 historical/joint_stage.py:fit。上面的输入约定与 retained_config.json 指明各阶段参数。完整历史驱动的顶层 main 还读取当次资产和统计目录，不能在缺少这些资产时直接执行。源码校验与单函数阅读不依赖这些大体积资产。

ENVIRONMENT.json记录整理时从已有执行环境元数据读取的依赖版本，requirements-recorded.txt不代表本轮新安装。原历史环境为Python3.12.14，PyTorch CUDA；求解器CPU float64。

## 来源

SimNL源代码来自 https://github.com/zhangce01/SimNL 的固定提交 a0871bcf31a607a7ec8acc99aff82f41ed7eaff3。上游原始文件与提交记录位于upstream/simnl，损失的可导入声明位于portable/losses.py。保留上游来源，不在此包为上游代码自行添加新许可证。当前本地副本未发现独立LICENSE文件，公开发布前需核对上游授权。

SOURCE_PROVENANCE.json记录实际执行源文件与当时冻结记录的匹配；SOURCE_MANIFEST.json固定本包文件哈希。包内不包含访问凭据，也没有执行外部上传。
