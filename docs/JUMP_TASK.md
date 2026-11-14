# 原地跳跃任务

`--task jump` 使用 v2 模型的实际关节绑定、四腿位置 PD 和两轮速度控制。
状态机与奖励参考旧工程 `state_machines/jump_takeoff.py` 的实测释放及落地恢复版本，
由电机和弹簧完成起跳。Actor 每帧 35 维、Critic 60 维、动作 6 维，普通 PPO 网络保持原配置。
当前场景是平面，速度与角速度目标均为零，先训练独立的原地跳跃。

## 控制流程和默认参数

参数在 `src/wheel_legged_lab/settings.py` 的 `JumpSettings` 与 `settings_for('jump')`。
状态机在 `components/jump_task.py`，奖励计算在 `components/rewards.py`。

| 阶段/参数 | 默认行为 |
| --- | --- |
| 站立/准备 | 站高 0.28 m，真实 episode 存活至少 2 s，双轮接触、姿态和速度稳定 |
| 自动请求 | 就绪后以 0.5/s 的概率速率触发，尝试结束后冷却 2 s |
| 峰值目标 | 机身基座距地面的绝对高度 0.40–0.44 m，不是轮底离地高度 |
| 蹬地 PUSH | 用弹道参考提供高度/速度奖励；释放高度 0.34 m |
| 实测释放 | 当前竖直速度 ≥ 参考释放速度的 60%（至少 0.3 m/s），并达到释放高度或双轮合力 ≤20 N |
| 提前离地释放 | 双轮净空 >2 cm、均无有效接触、合力 ≤20 N，且蹬地实测最大向上速度 ≥0.3 m/s |
| 蹬地超时 | 0.28 s 内未释放，记失败并恢复站立、冷却 |
| 真实腾空 | 双轮净空均 >2 cm、均无接触，连续 ≥0.04 s，蹬地最大向上速度 ≥0.3 m/s |
| 落地 LAND | 确认腾空后双轮接触，观测立即恢复普通站立模式；继续检查落地条件 |
| 稳定落地 | 站高误差 ≤0.06 m、竖直速度 ≤0.3 m/s、倾角约≤20°、角速度范数≤1.5 rad/s、平面速度误差≤0.4 m/s，持续0.2 s |
| 整次超时 | 1.5 s 内未完成稳定落地，记失败 |

落地成功还要求物理状态有限、只有轮子触地。摔倒和 episode 结束沿用原有终止规则，
正在进行的跳跃在终止时计失败，同一周期的落地事件也不能覆盖摔倒。
PPO 随机初始 episode 年龄不会跳过真实 2 s 的准备时间。
一次 Q 请求最多产生一次尝试；忙碌期间的重复请求不排队，空闲冷却期间的请求等待就绪。

## 奖励

下表权重由 `settings_for('jump')` 设置，其他站立/姿态/平滑性项继续生效。

| 奖励项 | 权重 | 含义 |
| --- | ---: | --- |
| `jump_push_velocity` | 1 | 超过0.3 m/s后的最大速度进展增量，每次尝试累计最多1；静止和重复速度不重复奖励 |
| `jump_push_height` | 0.25 | 弱高度参考跟踪 |
| `jump_airtime` | 3 | 真实腾空确认后的空中时间 |
| `jump_peak` | 50 | 已确认腾空的尝试退出时，奖励实际峰值接近目标 |
| `jump_takeoff` | 100 | 首次真实起跳事件 |
| `jump_success` | 200 | 稳定落地事件 |
| `jump_failed` | -25 | 蹬地超时、整次超时或尝试中摔倒 |

最终贡献乘控制周期0.02 s：起跳一次贡献2，稳定落地贡献4，失败贡献−0.5。
速度进展内部除以周期，累计贡献仍最多1，避免随控制频率改变。
PUSH/FLIGHT 暂停普通站高、竖直速度惩罚；LAND恢复。
PUSH/FLIGHT 的速度、偏航、直立正奖励减半，姿态负奖励保持，避免腾空时放任翻转。
LAND 的七维模式观测恢复 `[1,0,0,0,0,0,0]`，此时内部仍在检验落地，未提前计成功。

## 训练

从已验证的 v2 站立 actor 开始，critic、优化器和探索噪声重新初始化：

```bash
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python /home/rm-nuaa/wheel_leg/IasscLab/scripts/train.py \
  --task jump --strategy legacy_ppo \
  --checkpoint /home/rm-nuaa/wheel_leg/IasscLab/outputs/stance_alignment_stable_seed/stand/2026-11-12_11-07-30/model_199.pt \
  --transfer_actor --num_envs 1024 --max_iterations 3000 --headless
```

权重保存在 `logs/jump/时间戳/`。先学会0.28 m站高，再在就绪门槛内自动尝试跳跃。
迁移使用与来源相同的电机处理、动作顺序、PD、观测缩放/历史；有五维云台扩展的 spin actor
不能直接种到35维跳跃策略中。同一跳跃任务继续训练时使用跳跃权重，加 `--resume`，省略 `--transfer_actor`。
仅加载网络结构或完成短程 PPO 更新不代表已学会起跳。

## 评估和回放

训练后将 `JUMP` 替换为实际跳跃权重的绝对路径：

```bash
PY=/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python
JUMP=/absolute/path/to/trained_jump/model_3000.pt

$PY /home/rm-nuaa/wheel_leg/IasscLab/scripts/evaluate_jump.py \
  --checkpoint "$JUMP" --num_envs 32 --rounds 5 --headless --trace

$PY /home/rm-nuaa/wheel_leg/IasscLab/scripts/play.py \
  --checkpoint "$JUMP" --keyboard --num_envs 1 --steps 0
```

键盘 Q 请求一次跳跃。回放关闭自动跳跃，准备/冷却条件满足后才执行；W/S/A/D 在原地跳跃中禁用。
不用键盘时可通过 `--jump_at 3` 在回放3秒时请求一次跳跃。

评估默认每轮准备2.5 s、请求一次、再观测3 s，输出 `summary.json` 与可选 `trace.jsonl`。
分开报告请求、触发、实测释放、真实起跳、稳定落地和落地后存活到轮末的数量。
终止前的事件单独保留；原 episode 摔倒后，复位出的新 episode 不会冒充成功。
`--mode smoke --transfer_actor` 可用相容的 stand/walk 权重检查执行链路，此时报告不表示已训练跳跃。
smoke 仅将进入尝试的站高误差放宽到0.20 m，以便站立种子能进入状态机；
真实腾空和落地门槛不变。正式 evaluate 与训练均使用0.06 m的原门槛。
没有触发的 smoke 会明确报错，避免只运行普通站立就宣称跳跃链路通过。

TensorBoard 查看 `Episode/Jump/TakeoffPerAttempt`、`SuccessPerAttempt`、`AttemptCount`，
以及释放率、蹬地/无腾空/落地超时率和 `Landing*BlockedFraction`。
无腾空超时包含蹬地超时；各落地条件可能同时阻塞，比例不能相加。
TensorBoard 复位批次均值用于诊断，最终成功率使用独立试验评估。

跳跃控制参数、模式语义以 `jump_task` 字段写入权重契约，回放从权重恢复。
旧版未带该字段的跳跃权重仍使用保留的 `components/jump.py`；迁移到新任务须显式 `--transfer_actor`。

## 本次验证

62项CPU测试和静态检查通过；Isaac Sim中16个环境完成2次PPO更新，保存权重及优化器状态有限。
另做8环境×2轮的冒烟测试：16次请求均进入PUSH，随后按实测门槛蹬地超时退出，
全程动作、观测、奖励和状态有限，没有复位、真实起跳或稳定落地。
正式准备门槛下，这个只更新2次的站立种子还保持约0.36–0.40 m，没有达到0.28 m准备姿态。
因此本次验证了执行与失败处理链路，尚未产出学会跳跃的权重；真实释放、腾空、落地恢复和失败防误判由CPU回归覆盖。
记录位于 `outputs/jump_task_validation/report.json`，逐步轨迹在 `lifecycle_smoke/trace.jsonl`。
