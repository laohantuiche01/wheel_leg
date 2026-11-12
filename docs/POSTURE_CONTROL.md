# 站姿与云台 pitch 修正

针对 [model_500 诊断](STAND_DIAGNOSIS.md) 中的机身倾斜和云台 pitch 下垂，已做以下修改。
原 USD 几何、腿/轮控制增益、六维动作语义、观测维度、命令采样、跳跃时序和 PPO 参数保持不变。

## 机身姿态奖励

`settings.py` 的 `RewardSettings.weights` 新增：

| 奖励项 | 原始量 | 权重 |
| --- | --- | ---: |
| roll_error | roll²，rad² | -20 |
| pitch_error | pitch²，rad² | -10 |
| roll_pitch_rate | wx² + wy²，(rad/s)² | -0.2 |

以上三项只作用于 IDLE/LAND；保留原有 `upright` 和其余奖励，所有贡献仍乘控制步 dt。
机身保持水平由腿部策略学习完成，奖励修改不会直接改变既有网络的输出。

TensorBoard 新增 `Episode/Posture/`：`RollRMSDeg`、`PitchRMSDeg`、`TiltRMSDeg`、
`WheelHeightDiffRMSM`、`BothWheelsContactFraction`。轮心高度差不是髋高度差；直接髋高度差仍通过诊断脚本记录。
这些指标按实际采样时长统计，不用随机初始化后的回合计时反推测量时长。

## 云台 pitch 位置保持

`assets/v2_0.json` 新增 `auxiliary_holds.gimbal_pitch_joint`：

```text
目标关节角：0 rad
Kp：120 N·m/rad
Kd：3 N·m·s/rad
力矩限幅：±8 N·m
速度上限：6 rad/s
```

`AuxiliaryHoldController` 在每个物理子步计算并限幅力矩。该关节从辅助阻尼组中移除，
对应 Isaac 执行器设为零 PD，避免重复控制。yaw 仍为原来的被动阻尼。
此目标是关节相对零位，不是独立的世界系云台稳定器；机身倾斜仍需腿部控制修正。
位置 PD 没有积分/重力前馈，因此带载时允许小的静态角误差。

## 旧权重与微调

契约现在记录辅助保持参数。辅助控制不同的旧 v2 权重默认被拒绝；
`--allow_auxiliary_change` 只放行辅助控制差异，其余资产、动作方向、尺度和观测仍严格检查。
该选项不能与 `--resume` 同用，避免把控制变化当作完全相同训练的继续。

用旧 model_500 微调：

```bash
cd /home/rm-nuaa/wheel_leg/IasscLab
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python scripts/train.py \
  --task stand \
  --checkpoint logs/stand/2026-11-11_19-49-26/model_500.pt \
  --allow_auxiliary_change \
  --num_envs 1024 --max_iterations 1000 --headless
```

无需重新映射网络维度。新训练写入新的日志目录，原 checkpoint 和其契约不修改。
新生成的权重用 `play.py --task stand --checkpoint ...` 正常回放，无需兼容选项。

## 验证

- 11 项 CPU 测试通过，包括新姿态奖励、辅助 PD 和契约检查。
- 旧 model_500 加新 pitch 控制：4 个环境、500 步，0 次重置；双轮接触率 100%。
- 云台 pitch 均值从原来的 -30°变为约 -1.34°；仍是旧策略的机身倾斜，没有声称仅加位置保持就能纠正站姿。
- 对照数据：`outputs/stand_diagnosis_500/summary.json` 与 `outputs/stand_posture_hold_500/summary.json`。

已完成 256 环境、200 轮有限微调，写入独立目录，未覆盖原训练结果：
`outputs/posture_validation/stand/2026-11-11_20-20-00/model_199.pt`。
该编号表示新微调的第 0–199 轮，不是原训练的 model_199。

用同样 seed=42、4 个环境、500 步再次回放，排除前 2 秒后：

| 指标 | 原 model_500 | 新微调 model_199 |
| --- | ---: | ---: |
| roll RMS | 5.849° | 0.590° |
| pitch RMS | 2.228° | 1.199° |
| yaw 轴平均倾角 | 6.244° | 1.247° |
| 左右髋高差 RMS | 3.172 cm | 0.321 cm |
| 云台 pitch 均值 | -30° | -1.479° |
| 左/右轮接触占比 | 100% / 100% | 100% / 100% |
| 重置次数 | 0 | 0 |

左右不等高和机身倾斜明显减小，但不是严格零误差。新权重的 roll 瞬时范围约 -2.18°到 0.13°；
云台小幅静差符合纯 PD 在重力载荷下的行为。以上是有限站立回放，不证明跳跃、速度任务或扰动鲁棒性。
综合报告：`outputs/posture_validation/validation_summary.json`；逐步数据：`outputs/stand_posture_finetuned_199/trajectory.csv`。

可视化新权重：

```bash
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python scripts/play.py \
  --task stand \
  --checkpoint outputs/posture_validation/stand/2026-11-11_20-20-00/model_199.pt \
  --num_envs 1 --steps 10000
```
