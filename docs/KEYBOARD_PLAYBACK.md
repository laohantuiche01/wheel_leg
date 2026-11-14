# 统一键盘回放

`play.py --keyboard` 对所有任务启用 W/S、A/D、Q、Z/X。
先按照权重契约验证并加载模型，再启用独立的手动命令覆盖层；不会改写训练参数和权重。
省略 `--keyboard` 时仍使用原任务的固定指令和限制。

## 启动

本次已验证起跳和落地的v2跳跃权重：

```bash
/home/rm-nuaa/wheel_leg/wheeled-legged_RL/.venv/bin/python /home/rm-nuaa/wheel_leg/IasscLab/scripts/play.py \
  --checkpoint /home/rm-nuaa/wheel_leg/IasscLab/outputs/jump_fix_validation/dense_v2/jump/2026-11-14_15-25-43/model_199.pt \
  --keyboard --num_envs 1 --steps 0
```

也可将checkpoint替换为walk、spin、stand、height、velocity的权重，键位全部开放。
原来的任务名称、策略、历史长度、动作映射仍从权重恢复，不能任意换成不匹配的 `--task`。

点击 Isaac Sim 视口使窗口获得焦点。默认速度目标为零。
`--steps 0` 持续运行到关闭窗口或终端 Ctrl+C，原有回合超时和失败复位仍有效。
不能加 `--headless`。多个环境共享同一组键盘指令。

## 键位与范围

| 按键 | 行为 |
| --- | --- |
| W / S | 按住前进 / 后退，默认 ±2 m/s，松开请求零速 |
| A / D | 按住正向 / 反向旋转，默认 ±2 rad/s，松开请求零角速度 |
| Z / X | 升高 / 降低目标高度，默认0.20–0.38 m；松开保持 |
| Q | 请求一次跳跃，站稳并结束冷却后执行 |
| L | 清除速度和待执行跳跃请求，恢复初始目标高度 |

W/S与A/D可以组合；Q与高度指令也不再按任务名称禁用。
真实离地、姿态、接触、冷却和稳定落地判据继续执行，Q并不绕过这些条件。
同轴相反按键同时按下时目标为零，松开其中一个后恢复另一个目标。

幅度参数：`--keyboard_speed 2`、`--keyboard_turn_rate 2`。
高度参数：`--keyboard_height_rate 0.2`、`--keyboard_min_height 0.20`、`--keyboard_max_height 0.38`。
键盘范围独立于单任务训练命令范围；松开归零的是目标，速度指令仍按现有加速度限制变化。
跳跃目标按当前站高保留原来的释放/峰值相对高度；开始尝试后冻结这些参考，避免Z/X重写本次峰值。

控制入口全部开放不等于一个单任务权重已学会全部行为。
例如纯行走权重未训练原地跳跃，虽然Q会进入跳跃状态机，仍可能无法起跳；
跳跃权重也未必能稳定跟随2 m/s的移动或旋转指令。要同时稳定完成这些行为，需要相应的联合训练。

## 命令行初始值与焦点

带`--keyboard`时，`--vx`、`--wz`、`--command_height`作为初始目标；相应按键接管该轴，L清零速度。
Z/X可以修改命令行指定的初始高度。失焦也会清除初始速度、按键、短脉冲和排队的跳跃，保留已设高度。
正在进行的跳跃继续按物理状态完成，不会因为L或失焦突然重置姿态。

键盘模式自动实时运行并跟随机器人；`--no_camera_follow`启用手动相机。
不用键盘时可单独加 `--real-time`，固定CLI指令的原行为保留。

远程桌面/输入法的短促按下/松开或字符事件沿用默认0.2秒的脉冲兼容，后续脉冲续期，停止输入自动归零。
正常长按松开立即请求零速。通过`--keyboard_pulse_hold 0`可关闭兼容。

## 日志与验证

输入和命令记录到 `outputs/playback/keyboard_时间戳.jsonl`，可用`--keyboard_log`指定路径。
日志包含原始事件、统一控制标记、目标与实际速度、跳跃请求和阶段反馈。
终端会显示等待条件、PUSH、FLIGHT、LAND及失败原因。

75项CPU测试通过，覆盖所有任务的全键位、实际命令/跳跃路由、CLI接管、取消排队请求及配置不变性。
GUI验证使用真实play.py、已保存策略和原生Carb事件，逐一检查W/S/A/D/Z/X/Q/L到环境的目标指令。
记录在 `outputs/unified_keyboard_validation/`。这些是输入路由验证，不是未训练行为的成功率评估。

此前按任务限制键位的验证记录保留在`outputs/keyboard_validation/`，不代表当前统一键盘模式的键位限制。
