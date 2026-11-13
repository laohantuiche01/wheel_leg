# 行走速度微调实测

2026-11-12，从原行走训练的 `logs/walk/2026-11-12_14-32-26/model_800.pt` 开始。
原 checkpoint 保持不变。加载 Actor、Critic 和已有动作噪声，优化器重新初始化；
观测、隐藏层、六维动作、电机参数、命令范围及终止条件保持原值。

本报告记录的是历史命令范围：vx 为 ±1 m/s、wz 为 ±1.2 rad/s。
2026-11-13 现行行走默认课程为纯平移，vx ±2 m/s、wz=0；
严格复现本报告须先恢复历史命令配置，旧权重不能直接作为新范围任务的严格微调/续训权重。

## 执行的两轮训练

1. `outputs/walk_finetune/walk/2026-11-12_15-25-21`：1024 环境，300 次更新，
   学习率 3e-5，每批更新 3 次，原速度奖励容差 sigma=1。
   单独降低学习率未带来全面改善；后期速度/转向误差上升。
2. `outputs/walk_finetune_velocity/walk/2026-11-12_15-34-38`：相同微调设置，200 次更新，
   显式收紧速度奖励为 `exp(-(e_v/0.25)^2)`，保留原平方误差惩罚。
   后期速度误差较低，但持续倾斜更明显；未选最后一个 checkpoint。

各保存点在 seed=42 下做了 13 条固定指令、每条 2 个初始化、10 s 的初筛。
第 2 轮的 `model_50.pt` 在速度与姿态之间更均衡，先选定它，再用 seed=123 复核。
文件迭代号从 0 开始，因此该 checkpoint 是追加 51 次 PPO 更新后的网络。

## 复核结果

平面、无随机推力，13 条停车/正反向速度/组合转弯指令，每条 4 个初始化，每个权重运行 20 s。
稳态统计从命令渐变完成后 2 s 开始，排除复位当步和非有限状态。
以下平均误差是每条指令 RMS 的算术平均，不是全部样本混合后的 RMS。

| 指标 | 原 model_800 | 候选 model_50 |
| --- | ---: | ---: |
| 平均前向速度误差 RMS（m/s） | 0.16465 | 0.11595 |
| 平均转向角速度误差 RMS（rad/s） | 0.11815 | 0.13311 |
| 最大工况俯仰 RMS（°） | 7.440 | 5.114 |
| 失败终止 | 0 | 0 |
| 正常回合超时 | 52 | 52 |
| 目标 +0.5 m/s 的实际平均速度（m/s） | 0.4203 | 0.4622 |
| 目标 -0.5 m/s 的实际平均速度（m/s） | -0.5490 | -0.4780 |
| 目标 -1 m/s 的速度误差 RMS（m/s） | 0.1153 | 0.2090 |

前向速度误差平均降低 29.6%，转向误差平均增加 12.7%。
这是速度跟踪候选，尚未通过全范围验收：高速倒车和大角速度组合转弯仍有明显误差，
部分工况变差；+0.5 m/s 的剩余偏差约 0.038 m/s，也高于此前提出的 0.03 m/s 目标。
没有验证推力、复杂地形、命令反转过程、长期运行或实机表现。

35 项 CPU 测试与静态检查通过。两轮训练标量与保存的网络参数均未发现 NaN/Inf。
选中权重另做了 `play.py` 自动配置、100 步无窗口加载检查。

## 使用候选

```bash
cd /home/rm-nuaa/wheel_leg/IasscLab
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python scripts/play.py \
  --checkpoint /home/rm-nuaa/wheel_leg/IasscLab/outputs/walk_finetune_velocity/walk/2026-11-12_15-34-38/model_50.pt \
  --num_envs 1 --steps 10000 --vx 0.5 --wz 0
```

复现第二轮训练使用 `scripts/train.py --task walk --strategy legacy_ppo --finetune --velocity_sigma 0.25`，
源权重传原 model_800，`--num_envs 1024 --max_iterations 200 --headless`。
训练后仍需评估各保存点，不应直接选最后一个。

完整机器可读报告、逐工况结果与对比图保存在：

- `outputs/walk_finetune_velocity/review_2026-11-12_15-34-38/report.json`
- `outputs/walk_finetune_velocity/review_2026-11-12_15-34-38/comparison.png`
- `outputs/walk_finetune_velocity/holdout_2026-11-12_15-34-38/summary.json`
- 各训练目录中的 `finetune.json`、`task_settings.json`、`training_review.json`

后续应分开验证转向误差约束、实际速度观测及高速倒车训练，避免仅优化前向速度而牺牲其他工况。
