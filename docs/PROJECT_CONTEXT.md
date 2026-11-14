# 项目上下文与新对话上手

本页是当前工程导航，实验索引见 [PROJECT_STATE.json](PROJECT_STATE.json)。记录于2026-11-14，后续以源码、选定机器人/权重和新证据为准。

## 最短上手路径

```bash
cd /home/rm-nuaa/wheel_leg/IasscLab
python3 scripts/project_context.py
```

有具体权重/新模型时：

```bash
python3 scripts/project_context.py \
  --robot-spec /absolute/path/to/robot.json \
  --checkpoint /absolute/path/to/model.pt --json
```

不执行仿真、不导入工程、不加载pickle权重。输出默认/指定机器人、动作顺序与方向、源码导出的观测维度和时钟、相关证据及其文件/hash有效性。
checkpoint预检只比较可从spec读取的一部分物理/布局字段；完整加载必须经过`validate_checkpoint`或显式actor迁移。
返回码0表示快照可读取，3表示部分兼容预检发现差异，2表示输入或解析错误；0不等于已通过物理验证。
`--recent`只列最近日志权重，不推荐其中最大的编号。

本机训练Python提示：`/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python`。
它是既有环境的位置，不表示代码应改在旧仓库；换机器时先核实环境位置，不自动升级Isaac依赖。

## 当前代码与契约

| 内容 | 事实来源与当前值 |
| --- | --- |
| 默认机器人 | `settings.py:TaskSettings.robot_file`；当前`assets/v2_0.json` |
| 资产 | spec中的`usd`，当前`wheelbipe_v2_0.usd`；来源`model/v2_0.usdz` |
| 配置/绑定 | `robot.py`的`RobotDescription`、`RobotBinding` |
| 当前6动作 | 左leg1、左thigh、右leg1、右thigh、左wheel、右wheel |
| 策略方向 | 当前`[+1,+1,-1,-1,+1,-1]`，换模型重新量轴 |
| 腿 | 相对零位的位置目标；scale 0.5 rad、PD 60/2、限幅160 N·m |
| 轮 | 速度目标；scale 10 rad/s、Kd 0.2、限幅5 N·m |
| 执行 | `MotorBank`在物理子步算力矩，避免Isaac执行器重复PD |
| 时钟 | 物理0.005 s、decimation 4、控制0.02 s |
| 输入 | 当前A动作：Actor每帧17+3A，Critic30+5A；spin各加5；历史仅扩Actor |
| 当前维度 | 常规35/60，spin40/65，不能把所有权重当35维 |
| 默认策略 | `baseline`与`legacy_ppo`分别保留；回放从元数据恢复 |
| legacy_ppo | 动作EMA 0.8、有限差分速度EMA 0.2、每电机1–3物理子步延迟、观测延迟1控制帧 |
| 弹簧/云台 | 独立拥有者；pitch保持，spin世界yaw控制；具体参数读spec/settings |

这些数值是当前资产快照，不是下一个机器人的设计约束。完整非零零位、模式、方向和限幅以实际JSON为准。
当前测量/跳跃组件面向两轮、平地；USD不能直接作为MuJoCo模型使用。

## 代码改动导航

| 要修改的内容 | 先读 |
| --- | --- |
| 关节、轴向、零位、驱动归属 | `assets/*.json`、`robot.py`、`components/motors.py` |
| 复位、外力、质量 | `components/reset.py`、`disturbances.py`、`settings.py` |
| 速度范围、采样、静置和限速 | `settings.py`、`commands.py`、`locomotion.py` |
| 奖励/日志/测量坐标 | `rewards.py`、`state.py`；姿态/速度/位置要分清参考系 |
| 跳跃 | `jump_task.py`、`settings.py`的profile、`docs/JUMP_TASK.md` |
| 历史跳跃 | `jump.py`；不能把它与新的measured v1/v2混为一版 |
| 观测、历史、延迟、兼容 | `observations.py`、`legacy_control.py`、`checkpoints.py` |
| PPO/迁移/续训 | `ppo.py`、`runners.py`、`scripts/train.py` |
| 键盘与play | `play.py`、`playback_config.py`、`keyboard.py`、`keyboard_mode.py` |
| 仿真调度 | `env.py`，状态机同tick只推进一次，读取观测不推进时间 |

## 任务和可用能力

- stand：使用当前机器人校准rest height；height/velocity是既有通用任务，范围读profile。
- walk：独立前后平移，±2 m/s、wz=0；spin：独立原地旋转、±2 rad/s，云台世界yaw保持。
- jump：新训练默认measured_v2，站高0.28 m、机身峰值目标0.40–0.44 m。v1曾学成只站立；v2修复低速学习信号、探索和阶段编码。
- `--keyboard`：验证/加载权重后开放W/S/A/D/Q/Z/X，手动覆盖不改训练spec与权重；这不等于模型具有多任务能力。
- 当前没有本工程对应的MuJoCo/mjlab策略导出、目标模型和已验收sim2sim入口。

## 实验结论：按具体范围使用

| 已记录权重/功能 | 已有证据 | 已知局限 |
| --- | --- | --- |
| 独立walk model_2000 | 多数直行指令短期跟随；报告见索引 | −2 m/s过冲，4次中2次约18.8 s失败 |
| 独立spin model_1100 | ±1/±2 rad/s各20 s窗口无失败，云台定向良好 | 零指令不能视为定点停车；位置位移与速度积分有待核对 |
| jump v1 model_2999 | 正式32次都进入PUSH | 全部失败、0腾空，不能推荐为成功跳跃策略 |
| jump v2 model_199 | 两seed共64次正式短试验起跳/稳定落地，GUI Q成功 | 固定平面、短窗口；带噪训练落地率低于确定性回放 |
| 统一keyboard | walk/spin/jump GUI逐键路由，跳跃行为复测 | 只是控制入口，不是联合策略训练 |

具体相对路径、checkpoint hash和报告在`PROJECT_STATE.json`；可删的outputs丢失时不能继续宣称证据可复查。
新机器人不能直接继承这些结果。索引中的源码指纹是本次交接快照，不代表每份旧报告生成时的源码版本。

## 验证入口与范围

- `python3 scripts/static_check.py`：标准库静态检查；会更新`STATIC_CHECK.json`，不证明仿真成功。
- 既有环境运行`-m unittest discover -s tests -v`：CPU回归，当前数量以真实输出为准。
- `check_model.py`/`migrate_v2_model.py`：当前针对v2，不能自动验证/转换任意新模型。
- `evaluate_walk_suite.py`、`evaluate_spin_suite.py`、`evaluate_jump.py`：带权重的物理行为评估；先读参数与报告定义。
- `evaluate_jump.py --mode smoke`只放宽进入尝试的站高门槛，不代表正式门槛的策略效果；正式评估用evaluate。
- 上手/文档任务不自动启动Kit或长训；行为改动按授权范围做必要的小规模验证并记录实测。

换机器人详见[迁移流程](../skills/iassclab-maintainer/references/robot-change.md)。
使用`$iassclab-maintainer`可让新对话遵循同一上手流程；skill源与代码一同维护在`skills/iassclab-maintainer/`。
本机发现用的副本位于`~/.codex/skills/iassclab-maintainer/`；修改skill源后应同步安装副本，保留其用户定制元数据。
