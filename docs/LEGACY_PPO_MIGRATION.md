# 普通 PPO 控制与训练机制迁移（2026-11-12）

按用户选择，仅迁移普通 PPO 的共用机制。新代码独立放在 IasscLab，不导入旧项目、不修改 Isaac Lab 安装目录。
来源文件和 SHA-256、实际验证结果见工程根目录 `STRATEGY_MIGRATION.json`。

## 接入内容

| 旧机制 | 新实现与配置 |
| --- | --- |
| 动作低通 | `MotorSettings.action_filter_alpha`：当前动作权重，0.8 对应旧 0.2/0.8 EMA |
| 动作延迟 | 控制步和物理步两种时钟；物理步模式中腿/轮分组独立采样延迟，每回合固定 |
| 差分速度反馈 | `velocity_feedback="finite_difference"`；按实际物理 dt 计算，可选 EMA，轮子差分包角，复位/异常跳变回退到仿真速度 |
| 测量力矩曲线 | 单调速度点校验、分段线性插值、低速平台、高于末端速度时为零；与旧 M3508 一样对正负力矩对称限幅 |
| 旧 CSV 读取 | `scripts/import_motor_curve.py`；显式减速比和效率换算为输出轴 rad/s、N·m |
| Actor 历史与延迟 | `ObservationSettings.actor_history` / `actor_delay`；最旧到最新，延迟与堆叠长度独立 |
| 连续终止判据 | 倾斜连续达到 N 个控制步才终止；禁止接触、NaN/Inf 和过大关节/根速度立即终止 |
| 探索噪声下限 | 独立 `NoiseFloorRunner`，兼容 std 和 log_std 参数化，训练入口每轮更新后夹紧 |
| 动作平滑/能耗奖励 | 二阶动作差分平方、正向轮电功率的机械功率代理项；默认权重 0，需按任务标定 |
| 复位速度随机化 | 世界系根线速度/角速度范围，可选；所有关节仍使用 v2 的完整闭链零位 |

这里的 `wheel_power` 为正机械功率 `max(力矩×角速度, 0)` 的和，不包含电机效率/铜损，不是真实电功率测量。

历史观测每个控制步只移位一次；同一步重复读观测只替换当前帧。子集复位清空该环境缓存，避免旧回合历史泄漏。
差分速度只用于策略电机的 PD 反馈；Actor/真实 Critic 仍读取仿真速度，未冒充旧项目全部传感器估计链路。

## 两套配置

默认 `--strategy baseline` 保留现有站姿修正和六维动作，滤波关闭、延迟为零、单帧观测、立即普通终止。
已有 v2 站立权重按原命令正常使用；旧契约缺少的新默认观测字段按 1 帧历史、0 帧延迟解析。

`--strategy legacy_ppo` 是把旧普通 PPO 中可复用机制组合起来的 **v2 适配配置**，并非逐值复制某一个旧任务：

- 动作低通：0.8 当前 + 0.2 上次。
- 腿/轮动作延迟：每组、每环境采样 1–3 个 **物理步**，当前为 5–15 ms。
- 电机 PD 使用差分速度反馈，EMA 系数 0.2；旧权重回放恢复其保存的滤波值。
- Actor 观测延迟 1 个 **控制步**，当前为 20 ms。
- 倾斜终止持续 20 个控制步，当前为 0.4 s；禁止接触和数值异常立即终止。
- 探索 std 下限为 0.1；可用 `--noise_std_floor` 覆盖。
- 仍保留 v2 站高、腿/轮增益、机身 roll/pitch 奖励和云台 pitch 保持。

原项目普通 PPO 本身有多个参数组合：例如可选低通与差分执行器并非在每个旧任务都默认开启。
使用迁移配置意味着主动选择这些机制，需要重新训练验证，不能把它当成已有策略的等价运行环境。

## 训练与回放

```bash
cd /home/rm-nuaa/wheel_leg/IasscLab
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python scripts/train.py \
  --task stand --strategy legacy_ppo \
  --num_envs 1024 --max_iterations 3000 --headless
```

可选历史堆叠：加 `--actor_history 3`，Actor 输入变为 105 维；Critic 保持 60 维。
这仍是普通 MLP PPO，不是 HIM/DreamWaQ。

回放现在默认从权重同目录的 launch/schema 自动恢复策略和历史。也可显式提供一致的参数，例如：

```bash
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python scripts/play.py \
  --task stand --strategy legacy_ppo --actor_history 3 \
  --checkpoint /absolute/path/to/new_run/model_2999.pt --num_envs 1
```

实际权重文件名以训练输出为准。新电机参数会写入 checkpoint 契约；
改变历史长度、观测延迟、滤波或电机反馈后，旧的 baseline 权重会被拒绝。
`--allow_auxiliary_change` 只能放行辅助控制差异，不能绕过这些检查。
明确从相同模型、动作/历史维度的预训练权重迁移控制配置时，可在 train.py 使用 `--allow_control_change`；
它仅放行电机动作处理/反馈设置和观测延迟差异，不放行关节增益、动作方向、模型或历史维度的变化，也不能与 resume 同用。
新配置应独立训练；本次没有转换旧项目的网络权重。
回放退出和站姿错位的后续修正见 [PLAYBACK_STANCE_FIX.md](PLAYBACK_STANCE_FIX.md)。

## 测量曲线

旧配置引用的文件
`experiments/v14_flat/005_3508_motor/m3508_c620_current_closed_loop_raw_motor_shaft_curve_dense.csv`
在当前旧仓库中不存在。因此迁移了算法和读取工具，没有伪造测量数据或默认开启曲线。
拿到 CSV 后，按真实减速比转换；以下 `2` 仅是命令参数示例，不是该机器人的电机标定值：

```bash
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python scripts/import_motor_curve.py \
  --input_csv /absolute/path/to/measured.csv --output_json /absolute/path/to/wheel_curve.json \
  --group wheels --gear_ratio 2 --efficiency 1.0
```

训练和回放均加 `--motor_curves /absolute/path/to/wheel_curve.json`。
生成 JSON 的键是新模型 motor group 名字，点格式为 `[输出轴速度 rad/s, 输出轴力矩 N·m]`。
曲线与组额定力矩取更严格的限值。当前线性包络功能可另外开启，两者叠加时也取更严格值。

## 验证结果与边界

- 19 项 CPU 测试通过：滤波/延迟时钟、差分包角及复位、曲线单位与插值、历史顺序/重复读取/子集复位、持续终止与立即异常终止、std/log_std 下限及契约拒绝等。
- 新配置执行了 16 环境、5 轮 PPO 更新，Actor 105 维、Critic 60 维，损失与噪声统计为有限值；这只是执行和反向传播检查，随机初始化的新策略尚未训练到稳定站立。
- 默认配置回放原已验证 `model_199`：4 环境×500 步，0 次重置、双轮接触率 100%，roll RMS 0.590°、yaw 轴平均倾角 1.247°；与改动前对照数值一致。
- 旧项目的几何专用腿长 IK、整套奖励表及课程、粗糙地形、云台小陀螺、HIM/DreamWaQ 不在本次普通 PPO 共用机制迁移内。旧模型的这些部分不能仅靠替换关节名称用于 v2。

源代码仍保留唯一 `WheelLeggedEnv(DirectRLEnv)`，原 `CommandSampler`、`JumpController` 与 PPO MLP 配置文件未变。
