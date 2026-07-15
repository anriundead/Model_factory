# Skill 使用记录

> 项目根目录：`/home/a/Workspace/Model_factory/mergeKit_beta`
> Compose 根目录：`/home/a/Workspace/Model_factory`
> 容器内项目路径：`/app/ServiceEndFiles/Workspaces/mergeKit_beta`

本文记录本项目使用的 Codex skills 来源、安装路径和适用边界，避免与其他工作区或项目混淆。

## 已安装并用于模型服务门户

| Skill | 来源 | 本机路径 | 用途 |
|---|---|---|---|
| `frontend-design` | `anthropics/skills` -> `skills/frontend-design` | `/home/a/.codex/skills/frontend-design` | 新模型服务门户的主前端设计约束：视觉方向、排版、文案、非模板化设计。 |
| `prototype` | `mattpocock/skills` -> `skills/engineering/prototype` | `/home/a/.codex/skills/prototype` | 仅在需要比较多套门户布局时使用；原型必须是临时的，最终要删除或吸收到正式页面。 |
| `design-taste-frontend` | `Leonxlnx/taste-skill` | `/home/a/.codex/skills/taste-skill` | 作为前端视觉质量和 anti-slop 补充约束。 |
| `gsap-core` | `greensock/gsap-skills` | `/home/a/.codex/skills/gsap-core` | 指导 GSAP 动画 API、timeline-free 的基础动效。 |
| `gsap-performance` | `greensock/gsap-skills` | `/home/a/.codex/skills/gsap-performance` | 约束动画性能：只动画 `transform` 和 `opacity`，支持 reduced motion。 |
| `ponytail` | `DietrichGebert/ponytail` | `/home/a/.codex/skills/ponytail` | 控制实现复杂度，避免为门户引入不必要框架、构建链或抽象。 |

## Matt Pocock 仓库筛选结果

仓库：`https://github.com/mattpocock/skills`

已检查到的正式候选中，和本次前端门户最直接相关的是：

- `skills/engineering/prototype`：包含 UI 原型分支，适合在正式实现前比较多个门户布局。

未安装的相关项：

- `skills/deprecated/design-an-interface`：位于 `deprecated`，且关注模块接口设计，不作为本次 UI 设计工具。
- `skills/engineering/implement`、`skills/engineering/to-spec`：为用户显式调用型流程 skill，和当前已有 `writing-plans`、`executing-plans` 职责重叠，本轮不安装。
- `skills/engineering/codebase-design`：偏模块接口和代码结构设计，本轮前端门户不新增。

## 后续使用顺序

开发模型服务门户时按以下顺序使用：

1. `brainstorming`：确认页面边界、角色和交互意图。
2. `frontend-design`：确定门户的视觉系统和非模板化设计方向。
3. `design-taste-frontend`：执行视觉质量预检，避免 AI 默认设计痕迹。
4. `gsap-core` / `gsap-performance`：设计和约束动效实现。
5. `prototype`：仅当需要让用户比较多个布局方案时使用，且必须可删除。
6. `ponytail`：贯穿执行，优先复用当前 Flask 静态模板结构。

## 注意事项

- 新安装的 skill 可能需要重启 Codex 后才会出现在可用 skills 列表中。
- 本记录只适用于 `mergeKit_beta` 项目，不代表其他工作区的默认约束。
- 门户前端文件计划放在：
  - `mergeKit_beta/templates/model_gateway/console.html`
  - `mergeKit_beta/static/model_gateway/console.css`
  - `mergeKit_beta/static/model_gateway/console.js`
- 不把新门户逻辑写入旧页面的 `app.js`、`evaluation.js` 或全局 `styles.css`，除非只是添加导航入口。
