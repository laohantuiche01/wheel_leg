---
description: 仅用 Git 分析改动并自动生成 [TAG] 标签式 commit 信息；必要时建议拆分提交，默认运行本地 Git 钩子（可 --no-verify 跳过）
allowed-tools: Read(**), Exec(git status, git diff, git add, git restore --staged, git commit, git rev-parse, git config), Write(.git/COMMIT_EDITMSG)
argument-hint: [--no-verify] [--all] [--amend] [--signoff] [--tag <tag>]
# examples:
#   - /git-commit-tag                           # 分析当前改动，生成提交信息
#   - /git-commit-tag --all                     # 暂存所有改动并提交
#   - /git-commit-tag --no-verify               # 跳过 Git 钩子检查
#   - /git-commit-tag --tag ADD                 # 强制指定标签类型
#   - /git-commit-tag --amend --signoff         # 修补上次提交并签名
---

# Claude Command: Commit（[TAG] 标签格式，Git-only）

该命令在**不依赖任何包管理器/构建工具**的前提下，仅通过 **Git**：

- 读取改动（staged/unstaged）
- 判断是否需要**拆分为多次提交**
- 为每个提交生成 **[TAG] 标签式** 提交信息
- 按需执行 `git add` 与 `git commit`（默认运行本地 Git 钩子；可 `--no-verify` 跳过）

---

## Usage

```bash
/git-commit-tag
/git-commit-tag --no-verify
/git-commit-tag --all --signoff
/git-commit-tag --amend
/git-commit-tag --tag MOD
```

### Options

- `--no-verify`：跳过本地 Git 钩子（`pre-commit`/`commit-msg` 等）。
- `--all`：当暂存区为空时，自动 `git add -A` 将所有改动纳入本次提交。
- `--amend`：在不创建新提交的情况下**修补**上一次提交（保持提交作者与时间，除非本地 Git 配置另有指定）。
- `--signoff`：附加 `Signed-off-by` 行（遵循 DCO 流程时使用）。
- `--tag <tag>`：强制指定标签类型（如 `ADD`、`FIX`、`MOD` 等），覆盖自动判断。标签必须为大写字母。

> 注：如框架不支持交互式确认，可在 front-matter 中开启 `confirm: true` 以避免误操作。

---

## What This Command Does

1. **仓库/分支校验**
   - 通过 `git rev-parse --is-inside-work-tree` 判断是否位于 Git 仓库。
   - 读取当前分支/HEAD 状态；如处于 rebase/merge 冲突状态，先提示处理冲突后再继续。

2. **改动检测**
   - 用 `git status --porcelain` 与 `git diff` 获取已暂存与未暂存的改动。
   - 若已暂存文件为 0：
     - 若传入 `--all` → 执行 `git add -A`。
     - 否则提示你选择：继续仅分析未暂存改动并给出**建议**，或取消命令后手动分组暂存。

3. **拆分建议（Split Heuristics）**
   - 按**关注点**、**文件模式**、**改动类型**聚类（示例：源代码 vs 文档、测试；不同目录/包；新增 vs 删除）。
   - 若检测到**多组独立变更**或 diff 规模过大（如 > 300 行 / 跨多个顶级目录），建议拆分提交，并给出每一组的 pathspec（便于后续执行 `git add <paths>`）。

4. **提交信息生成（[TAG] 标签格式）**
   - 自动推断标签类型（`ADD`/`FIX`/`MOD`/`DEL`/`DOC`/`REF`/`TEST`/`CI`/`CHORE` …）。
   - 生成消息头：`[TAG] 简短描述`（首行 ≤ 50 字符，使用动词开头）。
   - **格式规则**：
     - 标签必须为**大写字母**，使用方括号包裹，如 `[ADD]`、`[FIX]`。
     - 标签与描述之间用**一个空格**分隔。
     - 简短描述使用动词开头（如"新增"、"修复"、"优化"、"删除"）。
   - 生成消息体（如有）：
     - 必须在 subject 之后空一行。
     - 使用列表格式，每项以 `-` 开头。
     - 每项**必须使用动词开头**（如"将…改为…"、"降低…"、"影响范围：…"）。
     - **禁止使用冒号分隔的格式**（如 ~~"Feature: description"~~、~~"Impl: content"~~）。
     - 说明变更的动机、实现要点或影响范围（3 项以内为宜）。
   - 生成消息脚注（如有）：
     - 必须在 Body 之后空一行。
     - **BREAKING CHANGE**：若存在破坏性变更，必须包含 `BREAKING CHANGE: <description>`。
     - 其它脚注采用 git trailer 格式（如 `Closes #123`、`Refs: #456`）。
   - 根据 Git 历史提交的主要语言选择提交信息语言。优先检查最近提交主题（例如 `git log -n 50 --pretty=%s`）判断中文/英文；若无法判断，则回退到仓库主要语言或中文。
   - 将草稿写入 `.git/COMMIT_EDITMSG`，并用于 `git commit`。

5. **执行提交**
   - 单提交场景：`git commit [-S] [--no-verify] [-s] -F .git/COMMIT_EDITMSG`
   - 多提交场景（如接受拆分建议）：按分组给出 `git add <paths> && git commit ...` 的明确指令；若允许执行则逐一完成。

6. **安全回滚**
   - 如误暂存，可用 `git restore --staged <paths>` 撤回暂存（命令会给出指令，不修改文件内容）。

---

## Best Practices for Commits

- **Atomic commits**：一次提交只做一件事，便于回溯与审阅。
- **先分组再提交**：按目录/模块/功能点拆分。
- **清晰主题**：首行 ≤ 50 字符，动词开头。
- **正文含上下文**：说明动机、方案、影响范围（禁止冒号分隔格式）。
- **遵循 [TAG] 格式**：`[TAG] 简短描述`。

---

## 标签定义

| 标签 | 说明 | 适用场景 |
| :--- | :--- | :--- |
| **[ADD]** | 新增功能、新文件或新资源 | 新功能开发、新增接口、新增文件 |
| **[FIX]** | 修复 Bug 或解决已知问题 | Bug 修复、紧急修复、安全修复 |
| **[MOD]** | 修改现有逻辑、优化代码或更新配置 | 功能调整、性能优化、配置变更 |
| **[DEL]** | 删除多余的代码、文件或资源 | 清理废弃代码、移除过时文件 |
| **[DOC]** | 更新文档（README, 接口文档, 注释等） | 文档编写、注释更新 |
| **[REF]** | 代码重构（不改变功能，仅优化结构） | 代码结构优化、抽离模块 |
| **[TEST]** | 新增或修改测试用例 | 单元测试、集成测试 |
| **[CI]** | 持续集成相关的配置更改 | CI/CD 流水线配置 |
| **[CHORE]** | 其他琐事（如更新 .gitignore, 依赖包升级等） | 依赖更新、工具配置 |

> 若传入 `--tag`，将**覆盖**自动推断。

---

## Guidelines for Splitting Commits

1. **不同关注点**：互不相关的功能/模块改动应拆分。
2. **不同类型**：不要将 `[ADD]`、`[FIX]`、`[REF]` 混在同一提交。
3. **文件模式**：源代码 vs 文档/测试/配置分组提交。
4. **规模阈值**：超大 diff（示例：>300 行或跨多个顶级目录）建议拆分。
5. **可回滚性**：确保每个提交可独立回退。

---

## Examples

**Good（简单提交）**

```text
[ADD] 新增用户登录接口
[FIX] 修复上传文件时的内存泄漏问题
[MOD] 调整搜索算法的权重分配
[DEL] 移除过时的 API 调试脚本
[DOC] 完善 SSH 端口配置说明
[REF] 抽离数据库连接池为独立模块
[TEST] 增加订单支付流程的单元测试
[CI] 更新构建流水线以支持 Node 20
[CHORE] 升级 lodash 到最新版本
```

**Good（包含 Body）**

```text
[MOD] 优化数据库查询逻辑

- 将原有的嵌套循环查询改为批量IN查询
- 降低了大约30%的数据库 CPU 占用率
- 影响范围：用户订单列表接口
```

```text
[ADD] 实现 OAuth2 第三方登录

- 接入 Google 和 GitHub 第三方登录
- 新增用户授权回调处理
- 优化登录状态持久化逻辑

Closes #42
```

```text
[FIX] 修复移动端按钮间距异常

- 调整按钮 padding 适配小屏设备
- 修复 iOS Safari 下的样式问题
- 优化触控目标尺寸
```

**Good（包含 BREAKING CHANGE）**

```text
[MOD] 重新设计认证 API

- 从 session 认证迁移到 JWT 认证
- 更新所有端点签名
- 移除已废弃的登录方法

BREAKING CHANGE: 认证 API 已完全重新设计，所有客户端需更新集成方式
```

**Split Example**

```text
- `[ADD] 新增支付方式类型定义`
- `[DOC] 更新新类型的 API 文档`
- `[TEST] 增加支付类型的单元测试`
- `[FIX] 修复新文件的 linter 告警` ←（如你的仓库有钩子报错）
```

---

## Important Notes

- **仅使用 Git**：不调用任何包管理器/构建命令（无 `pnpm`/`npm`/`yarn` 等）。
- **尊重钩子**：默认执行本地 Git 钩子；使用 `--no-verify` 可跳过。
- **不改源码内容**：命令只读写 `.git/COMMIT_EDITMSG` 与暂存区；不会直接编辑工作区文件。
- **安全提示**：在 rebase/merge 冲突、detached HEAD 等状态下会先提示处理/确认再继续。
- **可审可控**：如开启 `confirm: true`，每个实际 `git add`/`git commit` 步骤都会进行二次确认。
