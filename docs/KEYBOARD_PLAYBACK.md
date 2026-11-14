# 键盘回放

键位和按住/松开语义参考旧项目 `scripts/rsl_rl/keyboard_controller.py`。
使用 Omniverse 原生键盘事件，通过当前环境的公开命令接口发送目标；不导入旧工程或改动训练控制器。

## 启动

纯平移，W/S 对应 ±1 m/s：

```bash
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python /home/rm-nuaa/wheel_leg/IasscLab/scripts/play.py \
  --checkpoint /home/rm-nuaa/wheel_leg/IasscLab/logs/separate_motion/walk/2026-11-13_08-41-53/model_2000.pt \
  --keyboard --keyboard_speed 1.0 --num_envs 1 --steps 0
```

原地小陀螺，A/D 对应 ±2 rad/s：

```bash
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python /home/rm-nuaa/wheel_leg/IasscLab/scripts/play.py \
  --checkpoint /home/rm-nuaa/wheel_leg/IasscLab/logs/separate_motion/spin/2026-11-13_16-30-21/model_1100.pt \
  --keyboard --keyboard_turn_rate 2.0 --num_envs 1 --steps 0
```

点击 Isaac Sim 视口使窗口获得焦点。键盘模式启动时的速度目标为零。
`--steps 0` 持续运行到关闭窗口或在终端按 Ctrl+C；环境原有的回合超时和失败复位仍然有效。
不能与 `--headless` 同时使用。多个环境接收相同的键盘目标。

## 按键

| 按键 | 行为 |
| --- | --- |
| W / S | 按住前进 / 后退，松开请求零前向速度 |
| A / D | 按住正向 / 反向旋转，松开请求零角速度 |
| Z / X | 按住升高 / 降低目标高度，松开保持当前目标高度 |
| L | 清除键盘状态、恢复初始高度，不重置机器人姿态 |
| Q | 在 jump 任务请求一次原地跳跃；长按不重复触发 |

纯平移模型只接受 W/S，原地旋转模型只接受 A/D。
按键不会打开另一种运动轴。历史混合模型仍按保存的命令配置启用相应轴。
同轴相反按键同时按下时目标为零，松开其中一个后恢复另一个按键的目标。

按键幅度通过 `--keyboard_speed`（默认 2 m/s）和 `--keyboard_turn_rate`（默认 2 rad/s）设置，
会限制到权重保存的命令范围。松开归零的是目标；现有静置、加速度限制、电机延迟和策略响应保持原样。

远程桌面或输入法可能把一次长按转换成多组短促的“按下/松开”或字符事件。
这些事件如果在同一控制周期内到达，原先的按住状态会被立刻清空，造成按键有输入但目标一直为零。
现在对这种短脉冲保留默认 0.2 秒的指令，后续脉冲续期；停止输入后自动归零。
正常跨控制周期的长按仍在松开时立即请求零速。L 清除、窗口失焦及回合复位都会立即清除脉冲。
可通过 `--keyboard_pulse_hold 0.2` 调整兼容时间；设为 `0` 关闭短脉冲兼容。

高度变化速度由 `--keyboard_height_rate` 设置，默认 0.2 m/s，目标限制在训练时的站高范围。
当前纯平移和旋转模型的训练高度固定为 0.35 m，因此 Z/X 不改变其高度。
有可变高度范围的 height/velocity 模型可以使用 Z/X；Q 只对启用跳跃的 jump 模型生效。

失去窗口焦点会清除按键和待发送的跳跃请求，停止高度累加，保留已设高度。
回合自动复位时清除按键并恢复初始高度，需要重新按下运动键。

## 固定 CLI 指令与相机

沿用旧项目优先级：显式 `--vx`、`--wz`、`--command_height` 固定相应维度，优先于键盘。
想用键盘控制某一维度时，应省略该维度的固定参数。例如保留 `--vx 1` 时，W/S 和 L 不会解除该固定速度。
这些固定轴也不受失焦清除按键的影响。

键盘模式自动按仿真步长实时运行，并跟随第一个机器人，观察方向保持世界系固定，便于看云台定向。
加 `--no_camera_follow` 可改为手动相机；普通固定指令回放也可单独加 `--real-time`。

键盘功能只改变回放目标，未改动网络权重和电机 PD。已有的最大倒车速度失稳、停车漂移等问题仍需单独修正。

## 输入日志

键盘回放自动记录 `outputs/playback/keyboard_时间戳.jsonl`，启动时打印绝对路径。
也可用 `--keyboard_log /tmp/keyboard.jsonl` 指定路径。
日志记录原始按键/字符事件、窗口焦点、目标速度、加速度限制后的指令和实测速度，
静止期间每秒记录一次状态；可区分没有输入、目标被清零与策略没有跟随。

## 验证

49 项 CPU 测试通过，包含同周期按下/松开、字符脉冲续期、停止输入归零、原生长按松开和日志。
首版纯平移和小陀螺各在真实 GUI 的 `play.py` 中运行 200 步，
通过原生 Carb 事件缓冲注入按下、重复和松开事件，各通过 9 个目标值检查，退出时成功解除键盘订阅。
Q 请求及可变高度控制通过单元测试，尚未用训练好的跳跃/站高策略做 GUI 动作验证。
完整记录在 `outputs/keyboard_validation/report.json`；本次未修改已有权重。

短脉冲修复后的人工行走测试已确认可控制；目标 0.5 m/s 时，末三次稳定采样为
0.497、0.493、0.492 m/s。原地旋转在真实 GUI 中运行 270 步，同周期按下/松开脉冲、
超时归零、正常长按松开及 L 清除共 10 项检查通过。
修复证据和输入日志在 `outputs/keyboard_validation/pulse_fix/report.json`。
