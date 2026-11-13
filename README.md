# IasscLab · 下一版轮腿的 Isaac Lab 工程

这是从旧仓库拆出的独立重写版本，当前默认适配 **wheelbipe_v2_0 USD**；原 `wheelbipe_v1_0` 模型仍保留，可通过 `--robot_spec` 选用。
主环境只继承一次 `DirectRLEnv`，任务差异使用配置工厂，控制和扰动使用对象组合。
源码不导入旧 `agent_tasks`、`agent_world`、V13、V14，也不修改正在运行的旧训练。

**当前状态：已完成 v2 模型适配、姿态修正、普通 PPO 机制和纯平移/原地小陀螺任务，38 项 CPU 测试通过。**
原问题的实测指标见 [站姿诊断](docs/STAND_DIAGNOSIS.md)，修改内容与训练方式见 [姿态修正](docs/POSTURE_CONTROL.md)。
新迁移机制、配置和训练命令见 [普通 PPO 迁移说明](docs/LEGACY_PPO_MIGRATION.md)。默认 `baseline` 保留已验证行为；用 `--strategy legacy_ppo` 启用新适配配置。
回放会自动读取权重对应的策略/历史配置；退出原因和两腿前后错位的修正见 [回放与站姿修正](docs/PLAYBACK_STANCE_FIX.md)。
当前 `walk` 只训练前后平移（±2 m/s、wz=0），`spin` 只训练原地旋转（±2 rad/s、平移为零，云台保持世界 yaw）。
参数、Actor 迁移和命令见 [行走与小陀螺](docs/LOCOMOTION_TASKS.md)。
两任务已完成短时 PPO/回放验证，尚未训练到稳定跟踪；完整物理与扰动验证仍按 [后续验证清单](docs/VALIDATION.md) 进行。

## 代码结构

```text
src/wheel_legged_lab/
  env.py                    唯一环境类：只编排 DirectRLEnv 生命周期
  config.py                 唯一环境配置类 + 六种任务配置工厂
  settings.py               分组参数，不依赖仿真器
  robot.py                  机器人描述、Isaac 资产构建、关节和接触索引绑定
  assets/v2_0.json           默认机器人关节角色、静息位姿、电机参数
  assets/wheelbipe_v2_0.usd  从 model/v2_0.usdz 展平并重设 base_link 根的模型
  assets/v1_0.json           旧模型配置，供显式回退
  assets/wheelbipe_v1_0.usd  旧模型的独立副本
  components/
    motors.py               MotorBank / SpringController / AuxiliaryHoldController
    disturbances.py         MassRandomizer / ForceDisturbance
    commands.py             CommandSampler：站立、直行、转弯、原地旋转
    locomotion.py           新任务命令渐变、独立云台世界 yaw 控制
    jump.py                 JumpController：起跳、腾空、接触恢复、成功/超时
    state.py                StateReader：一次测量，供奖励和终止共享
    observations.py         ObservationBuilder：带版本的 Actor/Critic 观测契约
    rewards.py              RewardModel：奖励项和回合统计
    reset.py                ResetController：完整静息位姿复位
  ppo.py                    独立 PPO 配置
  checkpoints.py            保存/检查观测与机器人契约
scripts/
  train.py                  训练、恢复与显式 Actor 迁移
  play.py                   自动恢复配置、固定指令回放
  evaluate_locomotion.py    行走/小陀螺指标与轨迹导出
  static_check.py           仅 AST/JSON/TOML/文件校验，不导入工程或仿真器
  migrate_v2_model.py       从原始 USDZ 重建 v2 资产、配置和来源报告
  check_model.py            离线检查 USD 闭链、关节轴、零位和根坐标
tests/
  test_model_adaptation.py  CPU 检查索引、控制/观测方向、弹簧和 checkpoint
```

组件接口和调用顺序见 [架构说明](docs/ARCHITECTURE.md)。
旧版和新版本之间的行为变化见 [迁移说明](docs/MIGRATION.md)。

## 如何加入人为扰动

参数集中在 `settings.py`，例如在创建 `WheelLeggedEnvCfg(settings=settings)` **之前**：

```python
settings.mass.enabled = True
settings.mass.bodies = ("base_link",)
settings.mass.scale_range = (0.9, 1.1)
settings.mass.on_reset = False  # 默认启动时采样，避免每回合 CPU 物性写入
settings.force.enabled = True
settings.force.force_xyz = (30.0, 30.0, 0.0)  # 世界系 N，随机正负
settings.force.duration_s = (0.1, 0.3)
settings.force.interval_s = (5.0, 10.0)
settings.motors.delay_steps = (0, 2)          # 控制周期，每回合采样后固定
settings.motors.strength_range = (0.9, 1.1)
settings.motors.use_torque_speed_envelope = True
```

质量随机化始终从名义质量和惯量缩放，重复设置不会累计漂移。
外力在物理子步施加；结束、复位后写零清除。仿真中也可以直接调用：

```python
env.unwrapped.set_mass_scale(1.10, env_ids=[0])
env.unwrapped.apply_external_force((30.0, 0.0, 0.0), duration_s=0.2, env_ids=[0])
env.unwrapped.set_play_velocity_command(vx=1.0, wz=0.5, env_ids=[0])
env.unwrapped.request_jump_takeoff(env_ids=[0])  # 仅 jump 任务
```

## 电机控制策略

在当前机器人配置 `assets/v2_0.json` 中按电机组设置 `mode`：

- `position`：相对静息角的位置 PD，动作乘 `action_scale` 后加到静息角。
- `velocity`：目标速度 PD，动作乘 `action_scale` 后限制到速度范围。
- `torque`：动作直接映射为输出轴力矩。

PD、强度缩放、可选力矩-转速包络和最终限幅由 `MotorBank` 在**物理子步**统一处理。
Isaac 的策略电机执行器设为零 PD，仅接受力矩前馈，避免双重 PD。
v2 腿/轮沿用旧控制器增益和限幅：腿 60/2、160 Nm；轮 0/0.2、5 Nm。左右关节方向在配置中显式映射到策略坐标，电机延迟按控制步计算。
弹簧拥有独立控制器，不能再由其他组件重复写同一关节。
弹簧 `offset=None` 时取各自的模型复位位置，v2 为 0 m，v1 仍为 -0.0482 m；预载、刚度及阻尼沿用。
云台 pitch 使用独立位置保持（关节相对零位，Kp=120、Kd=3、限幅 8 N·m），不占用策略动作。
yaw 在 `spin` 中使用独立世界朝向 PD（20/1.5、限幅 2 N·m），其他任务仍为被动阻尼。
着地阶段增加 roll²、pitch² 和 roll/pitch 角速度平方惩罚，权重分别为 -20、-10、-0.2。

## 更换完整体模型

1. 复制一个机器人 JSON，把 `usd` 换成新的本地 USD（相对 JSON 所在目录）。当前默认配置为 `assets/v2_0.json`。
2. 填写完整静息位姿、机身/轮子/终止接触体名字，以及每个电机组的关节、模式、增益和限幅。
3. 主动、被动、弹簧、辅助关节必须互斥并覆盖 articulation 的全部关节。
4. 需要策略控制的云台关节放进 `motors`，同时从 `auxiliary_joints` 移除。
5. 非策略关节的执行器构建集中在 `RobotDescription.build_articulation()`；未来不同被动机构在这里扩展。
6. 使用 `--robot_spec /absolute/path/to/full_robot.json`。动作和观测维度由受控关节数量计算；stand 指令高度取该模型的 `rest_height`。
7. 新模型先按验证清单检查静态站姿、符号和接触索引，再训练。

当前测量和跳跃组件明确要求两轮、平地；下一步若增加复杂地形，先替换地面高度测量组件。

## 训练入口

沿用原有 Python/Isaac Lab 环境，不安装或升级依赖：

```bash
cd /home/rm-nuaa/wheel_leg/IasscLab
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python scripts/train.py \
  --task stand --num_envs 1024 --max_iterations 3000 --headless
```

`--task` 可取 `stand`、`height`、`velocity`、`jump`、`walk`、`spin`。
`--mass_randomization` 和 `--pushes` 分别开启启动质量随机化、随机力扰动。
同一新工程的 checkpoint 微调使用 `--checkpoint`，恢复优化器和迭代号另加 `--resume`。
保守微调加 `--finetune`：小学习率、3 次更新、每 50 轮保存并保留源权重动作噪声；
操作示例见 [任务说明](docs/LOCOMOTION_TASKS.md)。
已执行的两轮行走微调、候选权重及性能取舍见 [微调实测](docs/WALK_FINETUNE.md)。
**不要直接传旧项目 checkpoint**：虽然 Actor 仍为 35 维、Critic 为 60 维，v2 的动作关节和方向已变化，checkpoint 契约会拒绝旧权重。
`walk` / `spin` 可通过 `--checkpoint ... --transfer_actor` 迁移本工程匹配的 v2 Actor；
`spin` 逐帧扩展输入并重新初始化 Critic，命令见 [任务说明](docs/LOCOMOTION_TASKS.md)。
入口默认使用 v2；显式回退旧模型可加
`--robot_spec /home/rm-nuaa/wheel_leg/IasscLab/src/wheel_legged_lab/assets/v1_0.json`。
已有 v2 权重在观测/动作/资产完全匹配、仅辅助控制改变时，可用 `--checkpoint ... --allow_auxiliary_change`
显式微调；不能同时加 `--resume`。修正后的新权重按默认严格契约正常回放。

注册任务是 `WheelLegged-{Stand,Height,Velocity,Jump,Walk,Spin}-Direct-v0`。
包可直接由脚本引入，无需修改旧工程的 PYTHONPATH 或 editable 安装。

回放时通过 `scripts/play.py --checkpoint <权重路径> --task stand` 打开可视化窗口（不加 `--headless`）。
机器人目标高度使用 `--command_height 0.30`（单位 m），避免与 Isaac Sim 的 `height` 参数冲突；省略时沿用任务默认高度。

## 静态检查

```bash
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python scripts/static_check.py
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python scripts/check_model.py
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python -m unittest discover -s tests -v
```

第一条命令只使用标准库；第二条复用现有环境中的独立 USD 库；第三条使用 CPU PyTorch。
三者均不启动仿真器、不使用 GPU、不运行环境，结果不能替代仿真验证。

模型关节映射、转换记录与 SHA-256 见 `MODEL_MIGRATION.json` 和 `ASSET_PROVENANCE.json`；原仓库许可证保留在 `LICENSE`。
生成后模型的离线检查结果见 `MODEL_CHECK.json`。
