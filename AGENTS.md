# IasscLab 工程协作入口

本文件适用于本仓库。用户的当前目标和选定机器人优先于历史对话与旧实验说明。

## 新对话先建立上下文

在修改机器人、控制、任务、训练或回放行为前：

1. 阅读 [项目上下文](docs/PROJECT_CONTEXT.md)。
2. 运行 `python3 scripts/project_context.py`；有指定模型/权重时加 `--robot-spec <json>`、`--checkpoint <pt>`。命令只读，不启动仿真。
3. 阅读实际相关源码与权重旁的 `observation_schema.json`、`launch.json`、`task_settings.json`，确认本次机器人、控制/观测契约、任务版本和已有验证范围，再简要说明修改依据。

只做资料整理时不需要启动仿真。行为验证按改动风险选择；上手核查本身不扩成大规模训练或依赖升级。
终端若在旧 `wheeled-legged_RL` 仓库，先确认修改目标；不要把旧工程或空的mjlab目录当成本工程实现。

## 可复用 skill

源文件在 [iassclab-maintainer](skills/iassclab-maintainer/SKILL.md)，本机另安装到Codex用户skills目录。
换机器人时读 [迁移流程](skills/iassclab-maintainer/references/robot-change.md)；诊断控制/训练问题时读 [控制与证据](skills/iassclab-maintainer/references/control-and-evidence.md)。

## 具体工程约束

- 唯一环境类直接继承 `DirectRLEnv`，行为按组件组合。关节按语义名绑定，传感器索引独立解析。
- 机器人JSON、源码和checkpoint契约是事实来源；`docs/PROJECT_STATE.json`是带日期的证据索引。脚本中的资产预检不是完整兼容认证。
- 相同输入/输出维度不代表同一物理/观测含义。保留完整checkpoint校验；跨模型先建立具体映射，不能靠删校验或改元数据强行加载。
- 当前资产检查/迁移脚本含v2专用断言。换机器人先复核这些假设，不能用旧报告代替新模型验证。
- 不把静态检查、PPO冒烟、统一键盘或Isaac回放称为任务收敛或sim2sim完成。
- 已有的脏改动和实验保留；新模型/实验用独立名称和目录。行为变化后同步上下文和证据，不仅留在对话。

同名参数、默认模型、测试数量和推荐权重都可能随代码改变；先查当前状态，再使用文档中的示例。
