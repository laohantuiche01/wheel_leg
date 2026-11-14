# 原地跳跃任务

`--task jump` 使用 v2 模型的实际关节绑定、四腿位置 PD 和两轮速度控制。
状态机与奖励参考旧工程 `state_machines/jump_takeoff.py` 的实测释放及落地恢复版本，
由电机和弹簧完成起跳。Actor 每帧 35 维、Critic 60 维、动作 6 维，普通 PPO 网络保持原配置。
当前场景是平面，速度与角速度目标均为零，先训练独立的原地跳跃。

新训练默认使用 `measured_v2`。旧 `measured_v1` 权重仍从其契约恢复原来的奖励和模式编码；
跨版本训练使用显式 actor 迁移，重新初始化 critic、优化器和探索噪声。

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
| `jump_push_velocity` | 10 | 从0 m/s开始的实测最大向上速度进展增量，归一化后每次尝试最多1；静止和重复速度不重复奖励 |
| `jump_push_height_progress` | 4 | 相对本次起跳高度的实测抬升进展，目标为释放高度，每次尝试最多1 |
| `jump_push_height` | 0.25 | 弱高度参考跟踪 |
| `jump_airtime` | 3 | 真实腾空确认后的空中时间 |
| `jump_peak` | 50 | 已确认腾空的尝试退出时，奖励实际峰值接近目标 |
| `jump_takeoff` | 200 | 首次真实起跳事件 |
| `jump_success` | 300 | 稳定落地事件 |
| `jump_failed` | -50 | 蹬地超时、整次超时或尝试中摔倒 |

最终贡献乘控制周期0.02 s：起跳一次贡献4，稳定落地贡献6，失败贡献−1。
速度/抬升进展内部除以周期，每次尝试的原始进展各最多1，权重后的累计贡献分别最多10与4，避免随控制频率改变。
PUSH/FLIGHT 暂停普通站高、竖直速度惩罚；LAND恢复。
PUSH/FLIGHT 的速度、偏航、直立正奖励减半，姿态负奖励保持，避免腾空时放任翻转。
PUSH的力矩与动作变化惩罚缩放为原来的20%，腾空和落地恢复原值。
LAND 的七维模式观测恢复 `[1,0,0,0,0,0,0]`，此时内部仍在检验落地，未提前计成功。
v2的前五个模式标志为 `[站立,PUSH,FLIGHT,保留,跳跃中]`，将原先相同的蹬地/腾空观测分开；
仍保留两个峰值/时间字段，网络输入维度不变。旧策略迁移时，新启用的两列权重归零，避免激活未训练的随机系数。

## 训练

从已学会准备姿态的旧跳跃 actor 开始，critic、优化器和探索噪声重新初始化：

```bash
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python /home/rm-nuaa/wheel_leg/IasscLab/scripts/train.py \
  --task jump --strategy legacy_ppo --jump_profile measured_v2 \
  --checkpoint /home/rm-nuaa/wheel_leg/IasscLab/logs/jump/2026-11-14_00-26-43/model_2999.pt \
  --transfer_actor --num_envs 1024 --max_iterations 3000 --headless \
  --log_root /home/rm-nuaa/wheel_leg/IasscLab/logs/jump_v2
```

上述权重保存在 `logs/jump_v2/jump/时间戳/`。仍可从相容的站立actor开始，先学习0.28 m站高。
v2初始动作标准差为0.35，默认下限0.25；其他任务保持原值。
`--jump_profile measured_v1` 可复现旧训练配置，完整续训会自动恢复权重的版本；
用旧权重直接`--resume`不会自动切成v2。
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

键盘 Q 请求一次跳跃。回放关闭自动跳跃，准备/冷却条件满足后才执行。
`--keyboard`已统一开放W/S/A/D和Z/X，独立跳跃训练仍使用零速度指令；未训练的平移/旋转跟随能力需另行训练。
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

## model_2999 回放诊断

`logs/jump/2026-11-14_00-26-43/model_2999.pt` 的正式门槛评估（16环境×2轮、seed=42）：
32次请求全部进入PUSH，32次均失败，释放/真实腾空/稳定落地均为0。
PUSH采样中的最大向上速度约0.101 m/s，低于真实起跳所需的0.3 m/s。
训练3000次迭代的释放率、起跳率、成功率最大值均为0，最后100次迭代约98.6%的尝试发生蹬地超时。
人工回放日志也收到Q按键，问题来自策略尚未学会蹬地。

目前速度进展奖励在超过0.3 m/s后才出现，四腿探索标准差已降到约0.10；
这些记录提示需要补充低速蹬地的连续学习信号与课程，再重新训练，不能将站稳和高总奖励当成已学会跳跃。
本次检查保留训练奖励、门槛及权重，补充回放状态提示与JSONL中的`jump_request`/`jump_status`记录。
终端会显示PUSH、等待条件、失败时实测最大速度和释放速度门槛。
诊断结果在 `outputs/jump_review_2999/`。

## v2修改与训练对照

本次将PUSH速度进展从0 m/s开始累计，增加有上限的实测高度进展，
降低PUSH的力矩/动作变化惩罚，并将初始探索标准差改为0.35、默认下限改为0.25。
PUSH与FLIGHT使用不同观测标志。真实离地、落地、摔倒、电机PD和动作幅度保持原配置。

从旧`model_2999.pt`显式迁移actor，512环境训练200次更新，输出：
`outputs/jump_fix_validation/dense_v2/jump/2026-11-14_15-25-43/model_199.pt`。
69项CPU测试与静态检查通过；确定性评估使用正式门槛，seed=42和123各16环境×2轮，
共64次请求、64次释放、64次真实起跳、64次稳定落地并存活至轮末，复位0次。
机身峰值距地面约0.397–0.405 m，平均约0.401 m。
这是固定平面和短试验窗口的结果，不能推断长时间、复杂地形或外部扰动下的成功率。

真实GUI通过原生Q按下/松开事件，已验证PUSH→FLIGHT→LAND→IDLE，完成一次真实起跳和稳定落地。
训练采样含标准差≥0.25的随机动作，其落地率低于确定性回放，不能把两者混为同一个成功率。

回放本次新权重：

```bash
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python /home/rm-nuaa/wheel_leg/IasscLab/scripts/play.py \
  --checkpoint /home/rm-nuaa/wheel_leg/IasscLab/outputs/jump_fix_validation/dense_v2/jump/2026-11-14_15-25-43/model_199.pt \
  --keyboard --num_envs 1 --steps 0
```

点击视口、站稳后按Q。继续优化噪声下的落地，可从该v2权重同版本续训，
使用`--resume`、省略`--transfer_actor`，并显式选择`--noise_std_floor 0.1`。
对照记录在 `outputs/jump_fix_validation/report.json`。
