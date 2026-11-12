# 回放退出与两腿前后错位修正（2026-11-12）

## 回放退出

用户提供的权重为 `logs/stand/2026-11-12_10-14-55/model_800.pt`，对应训练记录是 `legacy_ppo`、单帧历史、
一帧观测延迟、差分电机反馈及 1–3 物理步动作延迟。
原回放命令未传 `--strategy`，旧脚本默认 `baseline`，导致契约检查拒绝加载。
此时 `finally: app.close()` 的 Kit 快速退出会先于 Python 异常展开，终端只留下关闭信息。

现在回放在启动仿真前检查权重路径、读取 `launch.json` 和 `observation_schema.json`，
自动选择任务、策略、历史和 seed，并恢复保存的电机反馈/延迟、弹簧及观测参数。
显式给出与训练不同的 strategy/history 时会在启动 Kit 前报清楚的错误。
模型资产、关节映射、控制增益等契约仍严格检查，不绕过不匹配。
main() 的异常现在先打印并刷新 traceback，再关闭仿真。

用户原命令可继续使用：

```bash
cd /home/rm-nuaa/wheel_leg/IasscLab
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python scripts/play.py \
  --task stand \
  --checkpoint logs/stand/2026-11-12_10-14-55/model_800.pt \
  --num_envs 1 --steps 10000
```

已用同一权重验证：自动解析为 legacy_ppo、Actor 历史 1、seed 42，成功加载网络并完成 1 环境、50 步回放。
`ManagerLiveVisualizer ... Manager does not exist` 是 DirectRLEnv 没有 manager 的提示；
外力 API 弃用、外力迭代选项和 TGS 迭代次数是其他警告，不是这次退出的契约错误。

## 两腿错位诊断

同一 model_800，在匹配的训练配置下回放 4 环境×500 步，统计第 2–10 秒：

- 两轮中心沿**机身 X 轴**的前后差 RMS：0.07658 m。
- 机身高度均值：0.14973 m，而站高指令为 0.37279 m。
- roll RMS：3.438°；pitch RMS：0.913°。
- 左轮接触占比约 84.85%，右轮约 27.68%。

所以这份权重学到的是低蹲、错位且支撑不稳定的姿态。仅修复回放配置不会改变已有网络的输出。
没有据此反转右侧电机方向或更改关节映射；这些映射已有模型轴和控制测试。

## 奖励修改

在 `RewardSettings.weights` 中：

- `height_error` 从 -20 调到 -80，加强高度平方误差惩罚。
- 新增 `wheel_stagger=-50`，惩罚两轮中心沿机身 X 轴的距离差平方。
- 新增 `double_support=2.0`，奖励双轮同时接地。
- 新增 `standing_wheel_clearance=-80`，惩罚站立时轮子悬空。
- 新增 `non_wheel_contact=-2`，惩罚轮子以外的刚体接触地面。

对齐、双轮支撑、站立轮底间隙三项仅在速度/转向指令绝对值都小于 0.1、且阶段为 IDLE/LAND 时生效。
行进、转向及 PUSH/FLIGHT 不被强制采用同一站姿。
非轮接触惩罚始终生效，避免用机身或腿杆撑地换取高度或对齐奖励。
不要求左右策略动作数值相等：右侧方向符号不同，应约束实际几何姿态。
新增 `Episode/Posture/WheelStaggerRMSM`，诊断 CSV 同时记录 `wheel_left_minus_right_body_x_m`。
已有网络需要微调或重新训练，修改权重配置不会直接纠正旧推理输出。

原权重和日志保持原样。有限微调验证写入独立 `outputs/stance_alignment_validation/`；
模型效果以随后回放为准，不能只用训练总奖励判断。

## 已拒绝的候选与进一步修正

从 model_800 直接微调 300 轮的候选 `outputs/stance_alignment_validation/stand/2026-11-12_10-49-21/model_299.pt`
虽然将前后错位减到约 1.8 cm，但回放出现单轮支撑、base 和 leg_5 接触地面及 18 次重置。
该候选未通过，不能作为修复模型。

由此将 base/gimbal 禁止接触改为**立即终止**；`duration_steps=20` 仅用于持续倾斜判据。
间歇碰地不再能通过每隔几帧清零连续接触计数来逃避失败。
同时增强双轮支撑、加入轮底间隙和非轮接触项。legacy_ppo 的差分速度 EMA 系数改为 0.2，
降低高频位置差分直接进入 PD 的影响；旧权重回放仍从其 schema 恢复原来的 1.0，不偷偷更改旧控制器。

后续从已验证的稳定站立权重进行控制配置迁移微调，结果写入 `outputs/stance_alignment_stable_seed/`。
`train.py --allow_control_change` 显式允许电机动作处理、速度反馈和观测延迟不同；
关节增益、模型、动作语义、历史维度仍严格检查，不能用于 `--resume`。

## 通过回放的候选

从原已验证站立权重 `outputs/posture_validation/stand/2026-11-11_20-20-00/model_199.pt`，
在修正后的 legacy_ppo 配置下进行 256 环境、200 轮微调，得到：
`outputs/stance_alignment_stable_seed/stand/2026-11-12_11-07-30/model_199.pt`。
它不是从失败的 300 轮候选继续训练，也没有覆盖用户 model_800。

4 环境×500 步、seed=42、无观测噪声，统计每环境第 2–10 秒：

| 指标 | 用户 model_800 | 新候选 |
| --- | ---: | ---: |
| 机身高度均值 | 0.1497 m | 0.3595 m |
| 机身 X 方向两轮错位 RMS | 7.658 cm | 2.002 cm |
| roll RMS | 3.438° | 0.025° |
| pitch RMS | 0.913° | 1.065° |
| 左/右轮接触占比 | 84.85% / 27.68% | 100% / 100% |
| 重置 | 0 | 0 |
| 非轮刚体接地 | 修改前未记录 | 0 |

新候选还有约 2 cm 前后错位、约 1.3 cm 站高误差，不宣称严格对称或已完成鲁棒性验证。
完整对照与源文件哈希见 `PLAYBACK_STANCE_FIX.json`；27 项 CPU 测试和静态检查通过。

可视化这份新权重：

```bash
cd /home/rm-nuaa/wheel_leg/IasscLab
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python scripts/play.py \
  --task stand \
  --checkpoint outputs/stance_alignment_stable_seed/stand/2026-11-12_11-07-30/model_199.pt \
  --num_envs 1 --steps 10000
```
