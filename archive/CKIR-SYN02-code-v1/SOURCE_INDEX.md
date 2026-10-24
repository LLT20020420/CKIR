# 固定源码定位索引

| 对应内容 | 文件 | 函数或类 | 起始行 |
|---|---|---|---|
| 完整重排序损失 | upstream/simnl/main.py | adaptive_reranking_loss | 233 |
| 交叉熵与默认零平滑 | upstream/simnl/utils.py | SmoothCrossEntropy | 148 |
| 文字残差零初始化 | historical/text_model.py | CKIRText.__init__ | 29 |
| 两次归一化与45/55混合 | historical/text_model.py | CKIRText.positive | 37 |
| 按类别轴计算文字一致性 | historical/text_model.py | CKIRText.consistency | 51 |
| 第一阶段实际训练 | historical/text_stage.py | fit | 53 |
| 第一阶段支持QP精修 | historical/text_stage.py | support_weights | 27 |
| W/b/gamma初始化 | historical/head_model.py | LPHead.__init__ | 7 |
| 第二阶段实际训练 | historical/aligned_head_stage.py | fit | 67 |
| 第三阶段实际联合更新 | historical/joint_stage.py | fit | 61 |
| L-BFGS-B与验收 | solvers/core.py | solve_box | 75 |
| 投影梯度残差 | solvers/core.py | projected_gradient | 50 |
| OSQP固定设置 | solvers/core_v04.py | BoxWorkspace._new_solver | 50 |
| OSQP每查询初始化及验收 | solvers/core_v04.py | BoxWorkspace.solve | 61 |
