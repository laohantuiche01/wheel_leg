# 行走与小陀螺任务

2026-11-12：新增 `walk` 和 `spin`，使用同一个 `DirectRLEnv` 和普通 PPO。
当前 `walk` 为两轮落地的纯前后平移，`spin` 为独立原地旋转，两种效果分别训练。
小陀螺按本次选择实现为**底盘原地持续旋转，云台保持初始世界系 yaw 朝向**。
当前地形仍为平面。

## 行为与控制

| 项目 | `walk` | `spin` |
| --- | --- | --- |
| 前向速度 | -2～2 m/s | 0 |
| 底盘转速 | 固定 0 | 随机正负，绝对值 1～2 rad/s |
| 命令采样 | 每 4～8 s；10% 站立，其余全部前后直行 | 每 30～40 s；默认 20 s 回合内保持方向 |
| 复位后命令 | 先静置 1 s，再按加速度限制渐变 | 相同 |
| 命令变化上限 | 线速度 0.8 m/s²，角速度 1.5 rad/s² | 相同 |
| 目标高度 | 0.35 m | 0.35 m |
| 云台 yaw | 原辅助被动阻尼 | 独立世界朝向 PD |
| 策略动作 | 原六维：四个腿位置目标、两个轮速度目标 | 相同 |
| Actor / Critic 输入 | 35 × 历史帧数 / 60 | 40 × 历史帧数 / 65 |

2026-11-13：先分开训练平移和原地旋转，不采样平移与旋转同时存在的指令。
新纯平移模型的手动回放会拒绝非零 wz，小陀螺会拒绝非零平移。
历史混合模型回放仍从其保存的配置恢复原范围；新课程的完整跟踪能力需要训练验证。
从旧范围的行走权重初始化新范围训练可使用
`--checkpoint ... --transfer_actor`，重新初始化 Critic 和优化器。
严格的 `--finetune` / `--resume` 要求命令配置也匹配，不会静默跨范围加载。

云台 yaw 控制在每个物理子步执行：

```text
error = wrap(target_world_heading - measured_world_heading)
torque = clip(20 * error - 1.5 * measured_world_yaw_rate, -2, 2) N·m
```

目标默认在每次复位后取初始世界朝向，也可以回放时指定固定角度。
控制器使用云台刚体的世界姿态和世界角速度；原 yaw 被动阻尼在 `spin` 中移除，避免两个控制器争用关节。
云台 pitch 继续使用原关节零位保持，Kp=120、Kd=3、限幅 8 N·m。
这是世界 yaw 定向，pitch 尚未改为世界水平稳定。

`spin` 的每帧观测在原 35 维后追加：相对 yaw 的 sin/cos、相对 yaw 速度、世界朝向误差的 sin/cos。
网络仍只控制底盘六个电机；新增观测使策略能感知云台运动与反作用。
隐藏层继续使用 256/128/64、ELU，PPO 参数和原命令采样、跳跃阶段算法保持原实现。

## 速度参考与奖励

v2 CAD 的 `base_link` 位于轮轴中点前方约 17 cm。原地旋转时该点必然绕轮轴运动。
新任务使用**左右轮轴位置中点的速度**，投影到机身水平朝向坐标系，跟踪前向速度、惩罚侧滑与旋转漂移。
底盘转速使用世界 Z 角速度，避免机身倾斜改变转速含义。原任务继续使用原速度参考。

两个任务均在整个着地阶段奖励双轮支撑、惩罚轮子悬空和两轮前后错位，保留高度、俯仰及触地保护。
当前 `walk` 的 roll 参考为零，并关闭转弯侧倾参考。
`spin` 额外惩罚轮轴中点平移，奖励云台世界朝向误差收敛。

参数入口均在 `src/wheel_legged_lab/settings.py`：

- `settings_for("walk")` / `settings_for("spin")`：各自命令范围与奖励权重覆盖。
- `CommandSettings`：静置时间、采样间隔、加速度限制与采样比例。
- `SpinSettings`：云台关节/刚体名称、PD、力矩/速度上限及观测缩放。
- `RewardSettings.weights`：通用奖励默认权重；任务工厂会覆盖其中部分值。

算法实现见 `components/locomotion.py`、`components/state.py`、`components/rewards.py`。
每次训练保存 `task_settings.json`，可核对该次训练的全部配置；回放从权重目录自动恢复控制和命令配置。

## 训练

以下从已验证的 v2 站立权重迁移 Actor，使用与源权重一致的 `legacy_ppo` 控制配置。
两个任务分别训练，输出到各自目录：

```bash
cd /home/rm-nuaa/wheel_leg/IasscLab
PY=/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python
STAND=/home/rm-nuaa/wheel_leg/IasscLab/outputs/stance_alignment_stable_seed/stand/2026-11-12_11-07-30/model_199.pt

$PY scripts/train.py --task walk --strategy legacy_ppo \
  --checkpoint "$STAND" --transfer_actor \
  --num_envs 1024 --max_iterations 3000 --headless

$PY scripts/train.py --task spin --strategy legacy_ppo \
  --checkpoint "$STAND" --transfer_actor \
  --num_envs 1024 --max_iterations 3000 --headless
```

从已有正式权重开始这两种独立课程，可以分别执行：

```bash
# 纯前后平移，日志独立保存在 logs/separate_motion/walk/
$PY scripts/train.py --task walk --strategy legacy_ppo \
  --checkpoint /home/rm-nuaa/wheel_leg/IasscLab/logs/walk/2026-11-13_01-05-51/model_700.pt \
  --transfer_actor --num_envs 1024 --max_iterations 3000 \
  --log_root logs/separate_motion --headless

# 纯原地旋转，日志独立保存在 logs/separate_motion/spin/
$PY scripts/train.py --task spin --strategy legacy_ppo \
  --checkpoint /home/rm-nuaa/wheel_leg/IasscLab/logs/spin/2026-11-13_00-33-20/model_1000.pt \
  --transfer_actor --num_envs 1024 --max_iterations 3000 \
  --log_root logs/separate_motion --headless
```

也可先训练 `walk`，再用其权重作为 `spin` 的 `--checkpoint ... --transfer_actor`。
已有 40 维小陀螺 Actor 也能通过该选项迁移到新的旋转范围，但云台观测和控制参数须一致。
迁移要求同一 v2 资产、动作方向、电机控制、观测缩放、延迟和历史长度；不能传旧项目或 v1 权重。
源权重使用多帧历史时，训练命令也要传相同的 `--actor_history`。

`--transfer_actor` 仅复制 Actor，Critic、优化器和探索噪声重新初始化。
转入 `spin` 时，逐帧复制原 35 列输入权重，新增 5 列置零；因此新增观测不会立即改变迁移后的策略均值。
迁移来源、哈希和列映射写入 `actor_transfer.json`。
训练中途恢复使用同任务 `--checkpoint ... --resume`，不能同时使用 `--transfer_actor`。
省略 checkpoint 和 transfer 参数可以从头训练。

同任务保守微调可用 `--checkpoint ... --finetune`：加载 Actor、Critic 和已有动作噪声，
重新初始化优化器，学习率 `3e-5`，每批更新 3 次，每 50 轮保存。
它不能与 resume、transfer 或控制变更例外同时使用；观测和控制契约仍严格检查。
不加 `--finetune` 的原入口行为保留。

```bash
$PY scripts/train.py --task walk --strategy legacy_ppo \
  --checkpoint /absolute/path/to/matching_walk/model.pt \
  --finetune --num_envs 1024 --max_iterations 300 \
  --log_root outputs/walk_finetune --headless
```

每次微调保存 `finetune.json`，记录来源哈希、实际学习率、初始动作噪声与优化器状态。
上述 `matching_walk` 权重的命令范围须与当前任务配置一致。
`max_iterations` 是本次追加的更新次数，新的微调日志从 0 编号。

行走微调可显式加 `--velocity_sigma 0.25`，将指数速度奖励从原来的 `exp(-e_v²)`
改为 `exp(-(e_v/0.25)²)`，保留平方误差惩罚。默认 sigma=1.0，原任务行为保持原值。
该值保存在 `task_settings.json` 与微调审计中；后续从该权重继续训练会继承，除非显式覆盖。
奖励容差不改变观测、动作或电机控制，评估应比较物理速度误差，不能直接横比不同容差的总奖励。

## 展示与测量

对正式训练得到的权重执行下列命令；任务、策略、历史长度会自动读取：

```bash
# 纯平移：默认 vx=0.5 m/s、wz=0
$PY scripts/play.py --checkpoint /absolute/path/to/walk/model_2999.pt \
  --num_envs 1 --steps 10000 --vx 0.5 --wz 0

# 小陀螺：默认原地 +2 rad/s；云台保持本回合初始世界朝向
$PY scripts/play.py --checkpoint /absolute/path/to/spin/model_2999.pt \
  --num_envs 1 --steps 10000
```

回放不加 `--headless` 即显示窗口。`spin` 可加 `--wz -2` 反转，
加 `--gimbal_heading 0` 将云台目标固定到世界 yaw=0 rad；它拒绝非零平移命令。
手动命令也经过静置和渐变，并在自动复位后继续生效。

需要记录速度跟踪、姿态、复位数、轮轴漂移与云台世界朝向时：

```bash
$PY scripts/evaluate_locomotion.py \
  --checkpoint /absolute/path/to/spin/model_2999.pt \
  --num_envs 4 --steps 1000 --headless \
  --output_dir outputs/spin_evaluation_001
```

输出目录必须是新目录，内含逐步 `trajectory.csv` 和 `summary.json`。
汇总排除每回合前 2 s、复位当步及非有限状态；复位次数单独计数。
`walk` 默认测试 0.5 m/s，`spin` 默认测试 2 rad/s，也可用 `--vx` / `--wz` 指定。

批量比较行走权重可使用 `scripts/evaluate_walk_suite.py --checkpoints <原权重> <微调权重> ...`，
加 `--output_dir <新目录> --headless`。当前纯平移模型默认测试 7 个停车/正反向速度指令，每条指令 4 个初始化，
每个权重运行 20 s，命令渐变结束后再等待 2 s 才统计稳态。
失败终止和正常超时分开计数；输出逐权重 JSON、完整轨迹 NPZ 和总表。
比较时应保持 seed、步数和 repeats 一致，选出候选后再用另一个 seed 复核。
加 `--extended_range` 后纯平移模型测试 11 个指令，覆盖正反向 1.5/2 m/s。
历史混合模型仍按其保存的配置测试 13/27 个指令，便于复核旧结果。
`evaluate_spin_suite.py` 自动按所测权重保存的角速度范围选择正反转测试点。

## 初次验证范围（历史记录）

35 项 CPU 测试通过，覆盖命令渐变、云台世界朝向/角度环绕、独占关节写入、轮轴速度参考、观测维度及多帧 Actor 迁移。
实际 Isaac Lab GPU 验证包括两个任务各 16 环境、8 次 PPO 更新，随后各 2 环境、250 控制步加载评估。
另外检查原站立权重和新 spin 权重的 `play.py` 自动配置与各 50 步回放。

这些验证证明执行与迁移链路可运行。8 次更新的测试权重仍有明显速度跟踪误差和复位，
不能视为已收敛的行走或小陀螺控制器。正式效果需完成训练后分别评估正反转、转弯与扰动。
本次未加入小陀螺世界系平移。完整记录见工程根目录 `LOCOMOTION_TASK_VALIDATION.json`。

## 当前独立课程验证

38 项 CPU 测试通过，包含纯平移/旋转的命令隔离、禁止手动混合命令、旧模型回放配置恢复、
既有 40 维小陀螺 Actor 的显式迁移以及云台控制不匹配时拒绝迁移。
两个独立课程均在 16 个环境完成了 8 次 PPO 更新，记录保存在 `outputs/separate_motion_validation/`。
纯平移/旋转分别完成 11/5 个指令的 6 s 加载评估；旋转短测仍有失败终止。
这些权重用于流程验证，不能视为新课程已收敛。完整记录在该目录的 `report.json`。

2026-11-13 对现有 8 次更新的小陀螺权重进行了多转速复测：执行链路可运行，
但未达到持续原地旋转要求。方法、失败数据与云台验证边界见 [小陀螺复测](SPIN_RETEST.md)。
