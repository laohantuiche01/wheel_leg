---
name: iassclab-maintainer
description: 维护 IasscLab 的轮腿 Isaac Lab 强化学习工程，先核实机器人、控制链、checkpoint 契约与实测证据，再修改任务、回放或迁移机器人。适用于本工程及其派生副本；普通 MuJoCo/Isaac 项目不自动套用本工程参数。
---

# IasscLab 维护与换机器人

目标：让新对话基于当前代码和选定机器人工作，而不是继承聊天中可能过时的模型名称、参数或成功率。

## 先取得本次上下文

定位用户指定的工程根目录，识别 `src/wheel_legged_lab`、`scripts`、`pyproject.toml`；不要因为终端位于旧 `wheeled-legged_RL` 就修改旧工程。

- 阅读工程的 `AGENTS.md` 与 `docs/PROJECT_CONTEXT.md`（如果存在）。
- 运行 `python3 <repo>/scripts/project_context.py`。用户指定模型或权重时分别加 `--robot-spec <json>`、`--checkpoint <pt>`；该命令只读，不导入 Isaac。
- 有关行为修改，再读对应源文件和 checkpoint 邻近的 `observation_schema.json`、`launch.json`、`task_settings.json`。快照的兼容预检只是部分比较，实际加载仍走 `validate_checkpoint` 或受控 actor 迁移。
- 修改前用简短进度说明确认：本次工程、机器人/资产、任务/策略、相关参数入口，以及已验证/尚未验证的范围。资料缺失时继续不依赖它的检查，只询问真正阻塞修改的信息。

若脚本/文档尚未随派生工程复制，直接检查 `settings.py` 的 `TaskSettings.robot_file`、机器人 JSON、`robot.py`、`env.py`、`observations.py`、`checkpoints.py`，建立同样的信息；不要默认为旧 v2 或跳过了解工程。

## 按任务补充阅读

- **换模型、改关节/驱动、旧权重加载新机器人**：读 [机器人迁移流程](references/robot-change.md)。
- **训练不收敛、回放没反应、控制/奖励/观测改动**：读 [控制与诊断约束](references/control-and-evidence.md)，再定位工程中的任务文档。
- **只改文档或查询当前状态**：只读核查即可，不自动扩成训练、依赖升级或资产重建。

## 维护边界

- 用户当前说明优先于历史记录。以源码、选定模型和权重契约为事实来源；`docs/PROJECT_STATE.json` 是带日期的证据索引，不是永远有效的默认模型/最优权重。
- 同维度不代表兼容。保留物理资产、动作顺序/方向、零位、控制处理、观测布局和任务版本的校验。需要迁移时先做具体映射，不能删校验、伪造 schema 或靠强行加载掩盖差异。
- 分清逻辑变更与物理参数变更。关节改名不自动意味着重调奖励、改轴向或改变已有调度；需要改变时说明证据和影响。
- 验证匹配修改范围：静态/CPU检查、模型检查、小规模物理回放、训练评估是不同层级。脚本成功或短 PPO 更新不等于学会任务；最新权重也不等于最好权重。
- `check_model.py`、`migrate_v2_model.py` 当前有 v2 专用假设，不能用它们的旧报告宣称新机器人通过。
- 行为或机器人切换后更新工程上下文和证据索引，记录具体模型、权重、参数、种子、时长及局限。源文件指纹只帮助发现变更；刷新指纹不能把旧实验升级成新验证。

## 交接结果

给出修改入口、为什么改、验证结果及尚未覆盖的范围。把后续仍需知道的结论写入工程文件，而不是只留在对话。
技能源码随仓库维护；更新skill后同步本机安装副本，并保留安装副本已有的用户定制policy/dependencies。
新对话可从以下请求开始：

> 使用 $iassclab-maintainer，先读取当前工程上下文并核对所选机器人和权重契约，再处理这次修改。
