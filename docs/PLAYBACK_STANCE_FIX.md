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
- 新增 `double_support=0.5`，奖励双轮同时接地。

后两项仅在速度/转向指令绝对值都小于 0.1、且阶段为 IDLE/LAND 时生效。
行进、转向及 PUSH/FLIGHT 不被强制采用同一站姿。
不要求左右策略动作数值相等：右侧方向符号不同，应约束实际几何姿态。
新增 `Episode/Posture/WheelStaggerRMSM`，诊断 CSV 同时记录 `wheel_left_minus_right_body_x_m`。
已有网络需要微调或重新训练，修改权重配置不会直接纠正旧推理输出。

原权重和日志保持原样。有限微调验证写入独立 `outputs/stance_alignment_validation/`；
模型效果以随后回放为准，不能只用训练总奖励判断。
