# Mergenetic 管理门户视觉优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变模型发布、密钥、调用、取消请求或 OpenAI 兼容接口的前提下，使 `/model-gateway` 成为清晰、克制、有即时交互反馈的管理员控制台，同时让 `/research` 保持纯用户工作区。

**Architecture:** 保持 Flask 模板、原生 CSS、原生 JavaScript 和既有 GSAP CDN。模板仅增加语义化的视觉包装与简短引导；`console.js` 保持所有 API 调用、DOM ID 和动态列表结构，最多扩展页面进入和状态表面的动画。CSS 是布局与交互的唯一来源，使用稳定网格和明确断点防止组件覆盖。

**Tech Stack:** Flask/Jinja 模板、原生 JavaScript、原生 CSS、现有 GSAP 3.12.5、Remix Icon、Python `unittest`、Node `--check`、Docker Compose。

## 设计与资料边界

- 使用已部署的 `brainstorming`、`writing-plans`、`ponytail`、`frontend-design`、`design-taste-frontend`、`gsap-core`、`gsap-performance`、`test-driven-development`、`verification-before-completion` 和 `requesting-code-review` skills；使用批次与路径补记到 `IMPLEMENTATION_SKILLS.md`。
- 借鉴明日方舟/女神异闻录式的“明确分区、强状态、短文案、离散反馈”交互语法，不复制角色、商标、插画、布局或源码。
- 外部检索只作为实现依据：MDN 的 `:focus-visible` / `prefers-reduced-motion` 文档和 GSAP `matchMedia()` 文档。GitHub 检索到的相关仿制库为 React/Tailwind 项目，维护度与技术栈不匹配，不引入、下载或复制其代码。
- 不新增框架、组件库、图片资产、数据库字段、API、队列任务、模型进程或 GPU 操作。

## 全局约束

- `/research` 不得出现 `/model-gateway` 链接、管理员跳转或 Admin Token 文案；品牌链接固定回 `/research`。
- `/model-gateway` 必须保留：`gateway-admin-panel`、`gateway-playground`、`gateway-developer-panel`、`gateway-create-service-form`、`gateway-create-key-form`、`gateway-chat-form`、`gateway-cancel-request-form`、服务/Key 列表 ID、所有请求状态 ID，以及 `/v1/chat/completions`、`/v1/requests/<request_id>`、`/v1/requests/<request_id>/cancel` 说明。
- 发布流程必须在发布区可见：选择模型 -> 填写服务配置 -> 管理员手动启动 -> 用 API Key 验证；OpenAI 兼容 cURL、Base URL、鉴权、usage、查询与取消说明必须保留。
- 所有交互元素具有 hover、active 和 `:focus-visible` 反馈；状态不单靠颜色表达。
- 仅动画 `transform` 与 `opacity`；无循环动画；`prefers-reduced-motion: reduce` 下不执行 GSAP 进入动画且 CSS 不保留位移/缩放过渡。
- 布局必须使用 `minmax(0, ...)`、`min-width: 0`、`overflow-wrap: anywhere` / `word-break` 和 1100px、720px 断点；不得用绝对定位承载正文、表单或操作按钮。唯一绝对定位仅限不遮挡内容的代码复制按钮与 toast。
- 不提交、不重置共享脏工作区中的无关变更。

## 文件职责

| 文件 | 本批职责 |
| --- | --- |
| `templates/model_gateway/research.html` | 去除管理员入口，收紧用户引导，保持用户任务与模式控件。 |
| `static/model_gateway/research.css` | 移除管理员链接相关样式，保证短文案和模式控件在桌面/移动端不拥挤。 |
| `templates/model_gateway/console.html` | 管理员控制台的语义分区、图标标题、可见发布流程和精简说明；保留全部功能 ID。 |
| `static/model_gateway/console.css` | 稳定响应式网格、视觉层次、状态表面、按钮/键盘反馈与 reduced-motion 覆盖。 |
| `static/model_gateway/console.js` | 保持请求行为不变；将现有页面进入动画收敛为可清理的响应式序列，并为动态状态表面提供一次性动画钩子。 |
| `tests/model_gateway/test_gateway_portal.py` | 模板、脚本和 CSS 的静态契约测试，防止功能 ID、用户边界和无障碍反馈倒退。 |
| `docs/model_gateway/IMPLEMENTATION_SKILLS.md` | 本批实际 skill 使用、路径与约束。 |
| `docs/model_gateway/ACCEPTANCE_20260714_ADMIN_PORTAL_VISUAL.md` | 验收命令、截图/人工检查、GPU 快照和回滚记录。 |

## Task 1: 先固定用户边界与现有管理员功能契约

**Files:**
- Modify: `tests/model_gateway/test_gateway_portal.py`

**Produces:** 对用户入口隔离、关键管理员 ID、OpenAI 兼容说明、发布流程和前端安全边界的失败测试。

- [ ] **Step 1: 写入失败测试**

添加 `_console_page()`、`_console_script()`、`_console_css()` 读取帮助函数，并添加以下断言：

```python
def test_research_page_has_no_administrator_navigation(self):
    page = self._research_page()
    self.assertNotIn('href="/model-gateway"', page)
    self.assertNotIn("管理控制台", page)
    self.assertIn('class="research-brand" href="/research"', page)

def test_console_keeps_operational_controls_and_visible_delivery_guidance(self):
    page = self._console_page()
    for token in (
        "gateway-create-service-form", "gateway-create-key-form", "gateway-chat-form",
        "gateway-cancel-request-form", "gateway-services-list", "gateway-api-keys-list",
        "gateway-copy-curl", "/v1/chat/completions", "/v1/requests/",
        "选择模型", "管理员手动启动", "API Key 验证",
    ):
        self.assertIn(token, page)
```

增加 CSS/脚本契约：存在 `:focus-visible`、`:active`、`prefers-reduced-motion`、`minmax(0,`、`min-width: 0`、`overflow-wrap`，以及 `window.gsap.matchMedia()` 与 `autoAlpha`。这些测试不检验像素，而是防止需求中的关键结构被后续改动删除。

- [ ] **Step 2: 验证红灯**

运行：

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest tests.model_gateway.test_gateway_portal.ModelGatewayPortalPageTestCase.test_research_page_has_no_administrator_navigation \
  tests.model_gateway.test_gateway_portal.ModelGatewayPortalPageTestCase.test_console_keeps_operational_controls_and_visible_delivery_guidance
```

预期：失败，因为当前研究页仍包含管理控制台链接，且当前模板没有明确发布流程文本。

- [ ] **Step 3: 记录不变量**

在 `IMPLEMENTATION_SKILLS.md` 添加本批名称、实际 skill 路径、无需 GPU/模型的约束和“不得更改 console.js 请求 URL/ID”的边界。

- [ ] **Step 4: 暂不修复，保留红灯证据**

将失败测试名称与简短输出写入验收文档草稿。不要在此任务修改运行代码。

## Task 2: 收紧研究工作区的用户引导

**Files:**
- Modify: `templates/model_gateway/research.html:12-16,34-46`
- Modify: `static/model_gateway/research.css`

**Produces:** 仅服务受邀用户的研究入口，不泄露管理员导航；短文案说明“直接提问”与“带资料研究”两种使用方式。

- [ ] **Step 1: 最小模板调整**

将品牌 `href` 从 `/` 改为 `/research`；删除 `research-admin-link` 元素。将标题下的支持文案替换为单句，例如“直接提问，或导入资料后获得可核验引用。” 保留模式按钮、来源、任务、结果、取消和 composer 的所有 ID。

- [ ] **Step 2: 清理对应 CSS，不扩大样式面**

删除仅用于 `.research-admin-link` 的规则和 reduced-motion 选择器引用。为工作区标题与模式选择器补充 `min-width: 0`、允许短文案自然换行的规则；不改变三种模式的数据和行为。

- [ ] **Step 3: 验证绿灯**

运行 Task 1 两个测试与：

```bash
node --check mergeKit_beta/static/model_gateway/research.js
git diff --check -- mergeKit_beta/templates/model_gateway/research.html mergeKit_beta/static/model_gateway/research.css mergeKit_beta/tests/model_gateway/test_gateway_portal.py
```

预期：全部退出码为 `0`。

## Task 3: 重组管理员模板的信息层级，不改动作或 ID

**Files:**
- Modify: `templates/model_gateway/console.html:52-300`

**Produces:** 可扫描的运维控制台结构，保留完整的发布、Key、验证、请求控制和开发接入能力。

- [ ] **Step 1: 将 Hero 收敛为操作入口**

保留 `gateway-hero`、两个锚点按钮、运行状态板及其统计 ID。标题与说明改为简短的管理员语气，例如“模型服务控制台”和“发布、启动、验证。”；不使用营销式大段说明。

- [ ] **Step 2: 增加可见的四步发布流程**

在 `gateway-admin-panel` 内、创建表单前或标题下增加一个非交互的 `gateway-publication-flow`。四步均使用 Remix icon、序号和短文字：选择模型、配置资源、手动启动、API 验证。此区域只解释现有流程，不新增自动启动或 GPU 分配。

- [ ] **Step 3: 标题使用图标与短标签**

为“管理员连接”“发布管理”“API Key”“用户调用”“请求控制”“开发接入”增加 `gateway-panel-title`/`gateway-panel-title-icon` 包装。保留现有 `h2`/`h3`、表单 ID 和刷新按钮 ID，避免 `console.js` 事件失效。

- [ ] **Step 4: 保留开发接入说明并提高可读性**

维持现有 cURL 内容、复制按钮 ID 和五项 guidance。为 Base URL、鉴权、usage、查询、取消加入图标/短标签；说明区域以“OpenAI 兼容接入”作为可见标题。不得将端点挪到隐藏抽屉、tooltip 或外链文档。

- [ ] **Step 5: 验证模板契约**

运行 Task 1 管理员页面测试；再执行：

```bash
grep -q 'id="gateway-create-service-form"' mergeKit_beta/templates/model_gateway/console.html
grep -q 'id="gateway-cancel-request-form"' mergeKit_beta/templates/model_gateway/console.html
grep -q '/v1/chat/completions' mergeKit_beta/templates/model_gateway/console.html
```

预期：均成功。

## Task 4: 实现无重叠的控制台视觉系统和交互反馈

**Files:**
- Modify: `static/model_gateway/console.css`
- Modify: `tests/model_gateway/test_gateway_portal.py`

**Produces:** 一致的精致运维控制台，能在 1440px、1024px、720px、390px 宽度下保持表单、列表、输出和操作按钮可用。

- [ ] **Step 1: 使用已有色板，移除装饰性噪音**

保留蓝色、墨色、绿色、橙色状态语义和当前中性表面；不新增品牌色、图片、渐变背景球或 3D 装饰。将背景与面板对比收敛到阅读优先的层次，让运行状态板成为唯一高对比信息面。

- [ ] **Step 2: 设定布局防撞规则**

为 `.gateway-page`、hero、栅格、面板、表单列、服务/Key 行、运行面板和请求控制增加或确认：`min-width: 0`、`minmax(0, ...)`、`overflow-wrap: anywhere`。桌面保留双列；`max-width: 1100px` 统一降为单列；`max-width: 720px` 让 inline field、按钮群、统计板和导航自然竖排。响应输出与代码块必须可横向/纵向滚动而不压住复制或取消按钮。

- [ ] **Step 3: 给所有可操作表面统一反馈**

覆盖顶栏链接、icon button、普通按钮、小按钮、复制按钮、服务行、Key 行、输入框和选择框。hover 只使用轻微 `translateY(-1px)`/阴影或边框变化；active 使用 `scale(.98)`；键盘通过 `:focus-visible` 提供不依赖颜色的 3-4px focus ring。禁用状态不得出现 hover lift。

- [ ] **Step 4: 保持状态可读**

扩展 `.gateway-status` 的 stopped/starting/failed/running 样式，并始终依赖动态文本 `status`。错误文本、模型名、白名单和 request ID 必须换行；停止按钮保持危险色且有“停止”文字，不替换成纯图标。

- [ ] **Step 5: 验证样式契约**

添加并运行测试，断言 CSS 含 `:focus-visible`、`:active`、`@media (prefers-reduced-motion: reduce)`、`min-width: 0`、`overflow-wrap: anywhere`、`.gateway-publication-flow`，并且模板具有该 flow 区域。

## Task 5: 限定 GSAP 为一次性、可清理的状态反馈

**Files:**
- Modify: `static/model_gateway/console.js:48-60,468-504`
- Modify: `tests/model_gateway/test_gateway_portal.py`

**Produces:** 当前已有的页面进入动画在响应式切换与 reduced-motion 下正确退出；动态服务、Key、模型数状态更新仅做一次性视觉确认，绝不改变数据、请求顺序或按钮可用性。

- [ ] **Step 1: 写入失败测试**

断言 `console.js` 包含 `window.gsap.matchMedia()`、`autoAlpha`、`reduced-motion`，并且入口动画只目标为 console 容器，而不修改 fetch URL、`requestJson` 或 `bindEvents` 中的 API 字符串。

- [ ] **Step 2: 最小实现**

保留 `runEntranceAnimation()` 的单次 `from` 序列；将 `matchMedia()` 实例保存于局部变量并让其在断点撤销时恢复内联状态。若添加 `animateStatusSurface`，只在 `setRuntimeSummary()` 发生实际文本变化且非 reduced-motion 时使用 `gsap.fromTo`/`gsap.to` 的 `autoAlpha`、`scale` 或 `y`，总时长不超过 0.24 秒。绝不为统计卡、状态点或 toast 设置重复动画。

- [ ] **Step 3: 防止动态渲染回归**

确认 `renderServices()`、`renderKeys()`、`renderUserModels()` 的 list ID、`data-start-service`、`data-stop-service`、状态文本和 `escapeHtml` 均保持。列表插入后的动画若实现，只按可见新行执行，且在 reduced-motion 下跳过。

- [ ] **Step 4: 验证**

运行：

```bash
node --check mergeKit_beta/static/model_gateway/console.js
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_portal
```

预期：通过；无需启动模型或使用 GPU。

## Task 6: 全量验收、人工视觉检查与记录

**Files:**
- Modify: `docs/model_gateway/ACCEPTANCE_20260714_ADMIN_PORTAL_VISUAL.md`
- Modify: `docs/model_gateway/IMPLEMENTATION_SKILLS.md`

- [ ] **Step 1: 记录非 GPU 基线**

```bash
git status --short
git diff --check
docker compose config --quiet
docker compose ps
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits
```

记录 GPU 2 的外部进程与显存状态，仅作基线；本批不得启动模型、融合、Ray 或 vLLM。

- [ ] **Step 2: 执行自动验收**

```bash
curl -fsS http://127.0.0.1:5000/healthz
curl -fsS http://127.0.0.1:5000/readyz
curl -fsS http://127.0.0.1:5000/research
curl -fsS http://127.0.0.1:5000/model-gateway
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests
node --check mergeKit_beta/static/model_gateway/research.js
node --check mergeKit_beta/static/model_gateway/console.js
git diff --check -- mergeKit_beta/templates/model_gateway mergeKit_beta/static/model_gateway mergeKit_beta/tests/model_gateway mergeKit_beta/docs/model_gateway
```

预期：HTTP 全部为成功响应、测试相对基线无新增失败、语法检查和 diff 检查均为 `0`。

- [ ] **Step 3: 浏览器人工验收**

在 1440px、1024px、720px、390px 宽度分别打开 `/research` 与 `/model-gateway`，检查：

1. `/research` 没有管理员跳转，品牌回到 `/research`，标题下只保留一条指导文案。
2. 控制台中标题、发布流程、服务列表、Key、调用/响应、取消、cURL 和 guidance 没有相互覆盖或裁剪。
3. Tab 键顺序可见；按钮/链接 hover、active、focus 均可辨；色弱或无色状态下可借文本理解服务状态。
4. 开启浏览器 reduced motion 后，没有位移、缩放或连续动画，但每一项操作仍可见、可点击。
5. 运行中/停止/失败服务以及长模型名、长 Key 白名单、长 request ID 使用本地模拟数据时换行，不溢出面板。

截图作为人工证据存入已忽略的 `logs/model_gateway/acceptance/<timestamp>/`，不上传 API Key、Admin Token、模型路径或真实响应内容。

- [ ] **Step 4: 发布验收记录**

写入实际命令、通过/失败结果、测试数量、截图视口、GPU 前后快照与未做的真实模型调用。记录已知前置告警不得被误归因于本批。

## 停止条件与回滚

- 任何模板契约、全量单测、HTTP smoke、Node 检查或 `git diff --check` 失败时，立即停止后续任务。
- 出现任一 ID 丢失、OpenAI 端点说明消失、研究页重新出现管理员导航、移动端遮挡、GPU 2 进程/显存状态变化、或浏览器发现键盘不可达时，判定本批不通过。
- 回滚范围严格限于：

```bash
git restore -- \
  mergeKit_beta/templates/model_gateway/research.html \
  mergeKit_beta/static/model_gateway/research.css \
  mergeKit_beta/templates/model_gateway/console.html \
  mergeKit_beta/static/model_gateway/console.css \
  mergeKit_beta/static/model_gateway/console.js \
  mergeKit_beta/tests/model_gateway/test_gateway_portal.py \
  mergeKit_beta/docs/model_gateway/IMPLEMENTATION_SKILLS.md \
  mergeKit_beta/docs/model_gateway/ACCEPTANCE_20260714_ADMIN_PORTAL_VISUAL.md
```

仅当上述文件本批之前没有用户未提交修改时才执行该命令；否则先导出本批 diff 到 `/tmp/mergenetic_admin_portal_visual_<timestamp>.patch`，再用反向 patch 回滚本批内容。不得触及 Gateway 数据、SQLite/PostgreSQL、Compose、Docker、Redis、ClamAV、模型目录或 GPU。

## Plan Self-Review

- 实现范围被限制在门户模板、CSS、现有 JS 的动画生命周期、静态契约测试与文档。
- 所有管理员动作和 API endpoint 都有保留断言，用户页与管理员页边界有负向断言。
- 防遮挡规则、四个视口、键盘、reduced-motion 和长文本均进入验收。
- 外部素材仅提供风格方向，不带入许可证或框架风险。
- 无真实推理、融合、GPU 或后台任务作为本批验收前提。
