(function () {
    const POLL_MS = 1500;
    const CONVERSATION_STORAGE_KEY = "mergeneticResearchConversationV1";
    const ACTIVE_SESSION_KEY = "mergeneticResearchArchiveActiveSessionId";
    const MAX_SESSION_ENTRIES = 60;
    const MAX_SESSION_BYTES = 256 * 1024;
    const MAX_MODEL_CONTEXT_MESSAGES = 16;
    const PENDING_FILE_STATES = new Set(["received", "queued_download", "queued", "scanning", "processing", "parsing"]);
    const FAILED_FILE_STATES = new Set(["failed", "rejected", "expired", "canceled", "dead_letter"]);
    const TERMINAL_JOB_STATES = new Set(["completed", "failed", "dead_letter", "expired", "canceled"]);
    const CANCELABLE_JOB_STATES = new Set(["queued", "running", "retrying", "paused_model_offline", "cancel_requested"]);
    const MergeneticResearchArchive = window.MergeneticResearchArchive;
    let researchTimeline = null;
    const state = {
        key: sessionStorage.getItem("mergeneticResearchKey") || localStorage.getItem("mergeneticResearchKey") || "",
        files: [],
        selectedFileIds: new Set(),
        turns: [],
        activeJobIds: new Set(),
        fileTimer: null,
        jobTimers: new Map(),
        lastSettingsTrigger: null,
        selectedModel: "",
        archive: MergeneticResearchArchive || null,
        archiveAvailable: Boolean(MergeneticResearchArchive),
        activeSessionId: localStorage.getItem(ACTIVE_SESSION_KEY) || "",
        activeSessionTitle: "新建研究",
        archiveSessions: [],
        pendingRenameSessionId: "",
        pendingDeleteSessionId: "",
        pendingDeleteMode: "",
        lastSessionRailTrigger: null,
    };

    const $ = (id) => document.getElementById(id);

    function motionAllowed() {
        return Boolean(window.gsap) && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    }

    function killResearchMotion() {
        researchTimeline?.kill();
        researchTimeline = null;
    }

    function runArchiveEntranceMotion() {
        killResearchMotion();
        if (!motionAllowed()) return;
        researchTimeline = window.gsap.timeline({ defaults: { ease: "power2.out", overwrite: "auto" } });
        researchTimeline
            .fromTo(".research-session-rail", { x: -14, autoAlpha: 0 }, { x: 0, autoAlpha: 1, duration: .24 })
            .fromTo(".research-command-strip", { y: -10, autoAlpha: 0 }, { y: 0, autoAlpha: 1, duration: .2 }, "<.04")
            .fromTo(".research-composer", { y: 12, autoAlpha: 0 }, { y: 0, autoAlpha: 1, duration: .2 }, "<.06");
    }

    function runSessionSwitchMotion() {
        killResearchMotion();
        if (!motionAllowed()) return;
        const entries = [...document.querySelectorAll(".research-chat-entry")].slice(-8);
        if (!entries.length) return;
        researchTimeline = window.gsap.timeline({ defaults: { ease: "power2.out", overwrite: "auto" } });
        researchTimeline.fromTo(entries, { y: 12, autoAlpha: 0 }, { y: 0, autoAlpha: 1, duration: .18, stagger: .025 });
    }

    function newId() {
        if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
        return `research-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function toast(message) {
        const node = $("research-toast");
        node.textContent = message;
        node.className = "research-toast show";
        clearTimeout(toast.timer);
        toast.timer = setTimeout(() => { node.className = "research-toast"; }, 2800);
    }

    function headers(json) {
        return Object.assign(
            { Authorization: `Bearer ${state.key}` },
            json ? { "Content-Type": "application/json" } : {},
        );
    }

    async function api(path, options) {
        const response = await fetch(path, options);
        const body = await response.json().catch(() => ({}));
        if (!response.ok) {
            const code = body.error?.code || "";
            const retryAfter = Number(response.headers.get("Retry-After") || 0);
            const quotaCopy = {
                chat_rate_limit_exceeded: "提问频率已到上限",
                research_submission_limit_exceeded: "本小时研究次数已到上限",
                research_concurrency_limit_exceeded: "已有研究任务正在进行",
                daily_import_quota_exceeded: "今日资料导入量已到上限",
            };
            const message = response.status === 429 && quotaCopy[code]
                ? `${quotaCopy[code]}${retryAfter ? `，请在 ${retryAfter} 秒后重试` : "，请稍后重试"}`
                : (body.error?.message || `HTTP ${response.status}`);
            const error = new Error(message);
            error.status = response.status;
            error.code = code;
            error.retryAfter = retryAfter;
            throw error;
        }
        return body;
    }

    function persistKey() {
        const target = $("research-remember-key").checked ? localStorage : sessionStorage;
        localStorage.removeItem("mergeneticResearchKey");
        sessionStorage.removeItem("mergeneticResearchKey");
        if (state.key) target.setItem("mergeneticResearchKey", state.key);
    }

    function normalizeFile(file) {
        return {
            id: String(file.id || ""),
            name: String(file.name || file.original_name || file.url || "未命名资料"),
            status: String(file.status || "received"),
            sourceKind: String(file.source_kind || file.sourceKind || "upload"),
            sourceUrl: typeof file.source_url === "string" ? file.source_url : "",
        };
    }

    function normalizeSource(source) {
        if (!source || typeof source !== "object") return null;
        const locator = source.locator && typeof source.locator === "object"
            ? source.locator
            : (source.kind && Number.isFinite(Number(source.value)) ? { kind: source.kind, value: Number(source.value) } : null);
        return { file_id: String(source.file_id || ""), locator, url: typeof source.url === "string" ? source.url : "" };
    }

    function normalizeTurn(turn) {
        const kind = turn?.kind || turn?.role;
        if (!turn || typeof turn !== "object" || !turn.id || !kind || typeof turn.content !== "string") return null;
        return {
            id: String(turn.id),
            kind: ["user", "assistant", "error"].includes(kind) ? kind : "error",
            content: turn.content,
            route: turn.route === "research" ? "research" : "direct",
            status: String(turn.status || "completed"),
            fileIds: Array.isArray(turn.fileIds) ? turn.fileIds.map(String) : [],
            model: typeof turn.model === "string" ? turn.model : "",
            jobId: turn.jobId ? String(turn.jobId) : null,
            idempotencyKey: turn.idempotencyKey ? String(turn.idempotencyKey) : null,
            citations: Array.isArray(turn.citations) ? turn.citations.map(String) : [],
            sources: Array.isArray(turn.sources || turn.locators) ? (turn.sources || turn.locators).map(normalizeSource).filter(Boolean) : [],
            replyTo: turn.replyTo ? String(turn.replyTo) : null,
            error: typeof turn.error === "string" ? turn.error : "",
        };
    }

    function snapshotBytes(snapshot) {
        return new TextEncoder().encode(JSON.stringify(snapshot)).length;
    }

    function isTerminalTurn(turn) {
        return turn.kind !== "user" || TERMINAL_JOB_STATES.has(turn.status) || turn.status === "failed";
    }

    function conversationSnapshot() {
        const snapshot = {
            version: 1,
            files: state.files.map(normalizeFile).filter((file) => file.id),
            selectedFileIds: [...state.selectedFileIds],
            turns: state.turns.map(normalizeTurn).filter(Boolean),
            selectedModel: state.selectedModel,
        };
        while (snapshot.turns.length > MAX_SESSION_ENTRIES) snapshot.turns.shift();
        while (snapshotBytes(snapshot) > MAX_SESSION_BYTES && snapshot.turns.length > 1) {
            const terminalIndex = snapshot.turns.findIndex(isTerminalTurn);
            snapshot.turns.splice(terminalIndex >= 0 ? terminalIndex : 0, 1);
        }
        if (snapshotBytes(snapshot) > MAX_SESSION_BYTES && snapshot.turns.length === 1) {
            snapshot.turns[0] = {
                id: snapshot.turns[0].id,
                kind: "error",
                content: "此条内容过长，无法在本次浏览器会话中恢复。",
                route: "direct",
                status: "failed",
                fileIds: [],
                model: "",
                jobId: null,
                idempotencyKey: null,
                citations: [],
                sources: [],
                replyTo: null,
                error: "session_entry_too_large",
            };
        }
        return snapshot;
    }

    function sessionTitleFromTurns(turns) {
        const question = turns.find((turn) => turn.kind === "user" && turn.content)?.content;
        return question ? question.replace(/\s+/g, " ").trim().slice(0, 48) : "新建研究";
    }

    function sessionRecordFromState() {
        return {
            id: state.activeSessionId,
            title: state.activeSessionTitle || sessionTitleFromTurns(state.turns),
            createdAt: state.activeSessionCreatedAt || Date.now(),
            updatedAt: Date.now(),
            messages: state.turns.map((turn) => ({ ...turn, role: turn.kind, locators: turn.sources })),
            sources: state.files.map((file) => ({ ...file, selected: state.selectedFileIds.has(file.id) })),
            selectedModel: state.selectedModel,
        };
    }

    async function persistActiveArchiveSession() {
        if (!state.archiveAvailable || !state.activeSessionId) return;
        const result = await MergeneticResearchArchive.save(sessionRecordFromState());
        if (!result.ok && result.reason === "session_limit") openSessionLimitDialog();
        if (result.ok) {
            const index = state.archiveSessions.findIndex((session) => session.id === result.session.id);
            if (index >= 0) state.archiveSessions[index] = result.session;
            else state.archiveSessions.unshift(result.session);
        }
    }

    function persistConversation() {
        if (state.archiveAvailable && state.activeSessionId) {
            persistActiveArchiveSession().catch(() => toast("本机记录暂时无法保存"));
            return;
        }
        sessionStorage.setItem(CONVERSATION_STORAGE_KEY, JSON.stringify(conversationSnapshot()));
    }

    function restoreConversation() {
        const raw = sessionStorage.getItem(CONVERSATION_STORAGE_KEY);
        if (!raw) return;
        try {
            const snapshot = JSON.parse(raw);
            if (!snapshot || snapshot.version !== 1 || !Array.isArray(snapshot.files) || !Array.isArray(snapshot.turns)) throw new Error("invalid session");
            state.files = snapshot.files.map(normalizeFile).filter((file) => file.id);
            const knownIds = new Set(state.files.map((file) => file.id));
            state.selectedFileIds = new Set((snapshot.selectedFileIds || []).map(String).filter((id) => knownIds.has(id)));
            state.turns = snapshot.turns.map(normalizeTurn).filter(Boolean).slice(-MAX_SESSION_ENTRIES);
            state.selectedModel = typeof snapshot.selectedModel === "string" ? snapshot.selectedModel : "";
            state.turns.forEach((turn) => {
                if (turn.route === "direct" && ["submitting", "running"].includes(turn.status)) {
                    turn.status = "failed";
                    turn.error = "页面已刷新，请重新发送";
                }
                if (turn.route === "research" && turn.jobId && !TERMINAL_JOB_STATES.has(turn.status)) state.activeJobIds.add(turn.jobId);
            });
        } catch (error) {
            sessionStorage.removeItem(CONVERSATION_STORAGE_KEY);
        }
    }

    function restoreStateFromSession(record) {
        clearTimeout(state.fileTimer);
        state.jobTimers.forEach((timer) => clearTimeout(timer));
        state.jobTimers.clear();
        state.activeJobIds.clear();
        state.files = (record.sources || []).map(normalizeFile).filter((file) => file.id);
        state.selectedFileIds = new Set((record.sources || []).filter((file) => file.selected && file.status !== "expired").map((file) => String(file.id)));
        state.turns = (record.messages || []).map(normalizeTurn).filter(Boolean).slice(-MAX_SESSION_ENTRIES);
        state.selectedModel = typeof record.selectedModel === "string" ? record.selectedModel : "";
        state.activeSessionId = String(record.id);
        state.activeSessionTitle = String(record.title || sessionTitleFromTurns(state.turns));
        state.activeSessionCreatedAt = Number(record.createdAt) || Date.now();
        state.turns.forEach((turn) => {
            if (turn.route === "direct" && ["submitting", "running"].includes(turn.status)) {
                turn.status = "failed";
                turn.error = "页面已刷新，请重新发送";
            }
            if (turn.route === "research" && turn.jobId && !TERMINAL_JOB_STATES.has(turn.status)) state.activeJobIds.add(turn.jobId);
        });
    }

    async function refreshArchiveSessions() {
        if (!state.archiveAvailable) return;
        state.archiveSessions = await state.archive.list();
        renderArchiveSessions();
    }

    async function revalidateArchiveSources() {
        if (!state.key || !state.files.length) return;
        await rehydrateFiles();
    }

    async function switchArchiveSession(sessionId) {
        if (!state.archiveAvailable) return;
        const record = await state.archive.read(sessionId);
        if (!record) return;
        restoreStateFromSession(record);
        localStorage.setItem(ACTIVE_SESSION_KEY, state.activeSessionId);
        await revalidateArchiveSources();
        renderAll();
        await refreshArchiveSessions();
        if (typeof runSessionSwitchMotion === "function") runSessionSwitchMotion();
        resumePendingTurns().catch((error) => toast(error.message));
    }

    async function createArchiveSession() {
        if (!state.archiveAvailable) return;
        const record = {
            id: newId(),
            title: "新建研究",
            createdAt: Date.now(),
            updatedAt: Date.now(),
            messages: [],
            sources: [],
            selectedModel: state.selectedModel,
        };
        const result = await state.archive.save(record);
        if (!result.ok) return openSessionLimitDialog();
        await switchArchiveSession(record.id);
    }

    async function initializeArchive() {
        if (!state.archiveAvailable) {
            restoreConversation();
            return;
        }
        try {
            await state.archive.open();
            let record = state.activeSessionId ? await state.archive.read(state.activeSessionId) : null;
            if (!record) {
                const existing = await state.archive.list();
                record = existing[0] || null;
                if (!record) {
                    const raw = sessionStorage.getItem(CONVERSATION_STORAGE_KEY);
                    if (raw) {
                        const migratedId = await MergeneticResearchArchive.migrateLegacy(JSON.parse(raw));
                        if (migratedId) {
                            sessionStorage.removeItem(CONVERSATION_STORAGE_KEY);
                            record = await state.archive.read(migratedId);
                        }
                    }
                }
            }
            if (record) await switchArchiveSession(record.id);
            else await createArchiveSession();
        } catch (error) {
            state.archiveAvailable = false;
            restoreConversation();
            toast("本机记录不可用，本次会话不会长期保存");
        }
    }

    function selectedSourceIds() {
        return state.files.filter((file) => state.selectedFileIds.has(file.id)).map((file) => file.id);
    }

    function findTurn(turnId) {
        return state.turns.find((turn) => turn.id === turnId);
    }

    function sourceState(fileIds) {
        const files = fileIds.map((id) => state.files.find((file) => file.id === id));
        if (!files.length || files.some((file) => !file || FAILED_FILE_STATES.has(file.status))) return "failed";
        if (files.some((file) => PENDING_FILE_STATES.has(file.status))) return "pending";
        return files.every((file) => file.status === "ready") ? "ready" : "pending";
    }

    function sourceStatusLabel(status) {
        const labels = {
            received: "已接收",
            queued_download: "等待下载",
            queued: "等待处理",
            scanning: "安全检查中",
            processing: "处理中",
            parsing: "解析中",
            ready: "已就绪",
            failed: "处理失败",
            rejected: "已拒绝",
            expired: "已过期",
            canceled: "已取消",
            dead_letter: "处理失败",
        };
        return labels[status] || status;
    }

    function statusText(turn) {
        if (turn.status === "preparing") return "正在准备资料";
        if (turn.status === "submitting" || turn.status === "queued") return "已加入研究队列";
        if (turn.status === "running") return "正在研究";
        if (turn.status === "paused_model_offline") return "模型服务暂停，等待管理员启动";
        if (turn.status === "cancel_requested") return "正在取消";
        if (turn.status === "canceled") return "已取消";
        if (turn.status === "failed") return turn.error || "任务未完成";
        return "";
    }

    function locatorLabel(locator) {
        if (!locator || typeof locator !== "object") return "来源定位不可用";
        const labels = { page: "第", slide: "第", paragraph: "第", web_title: "网页标题", web_paragraph: "网页第", web_table: "网页表格", web_figure: "网页图注", web_image_context: "网页图片说明" };
        const units = { page: "页", slide: "张幻灯片", paragraph: "段", web_paragraph: "段" };
        if (["web_title", "web_table", "web_figure", "web_image_context"].includes(locator.kind)) return `${labels[locator.kind]} ${locator.value}`;
        return labels[locator.kind] ? `${labels[locator.kind]} ${locator.value} ${units[locator.kind]}` : "来源定位不可用";
    }

    function appendEvidenceIndex(turn, message) {
        if (!turn.citations.length && !turn.sources.length) return;
        const evidence = document.createElement("section");
        const heading = document.createElement("strong");
        evidence.className = "research-evidence-index";
        heading.textContent = "证据索引";
        evidence.append(heading);
        if (turn.citations.length) {
            const citations = document.createElement("p");
            citations.textContent = `引用 ${turn.citations.map((item) => `[S${item}]`).join(" ")}`;
            evidence.append(citations);
        }
        turn.sources.forEach((source, index) => {
            const details = document.createElement("details");
            const summary = document.createElement("summary");
            const copy = document.createElement("p");
            const file = state.files.find((item) => item.id === source.file_id);
            summary.textContent = `${index + 1}. ${locatorLabel(source.locator)}`;
            copy.textContent = file ? file.name : "资料定位已保留";
            details.append(summary, copy);
            if (source.url) {
                const link = document.createElement("a");
                link.href = source.url;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                link.textContent = "打开网页来源";
                details.append(link);
            }
            evidence.append(details);
        });
        message.append(evidence);
    }

    function appendChatEntry(turn) {
        const timeline = $("research-chat-timeline");
        const message = document.createElement("article");
        const label = document.createElement("span");
        const entry = document.createElement("div");
        message.className = `research-chat-entry research-chat-${turn.kind}`;
        if (turn.route === "research" && !TERMINAL_JOB_STATES.has(turn.status)) message.classList.add("research-pending-entry");
        label.className = "research-chat-role";
        label.textContent = turn.kind === "user" ? "你" : turn.kind === "error" ? "状态" : "Mergenetic";
        entry.className = "research-chat-content";
        entry.textContent = turn.content;
        message.append(label, entry);
        const status = statusText(turn);
        if (status) {
            const stateLine = document.createElement("div");
            stateLine.className = "research-turn-status";
            stateLine.textContent = status;
            message.append(stateLine);
        }
        if (turn.jobId && CANCELABLE_JOB_STATES.has(turn.status)) {
            const cancel = document.createElement("button");
            cancel.className = "research-cancel-button";
            cancel.type = "button";
            cancel.dataset.cancelTurn = turn.id;
            cancel.disabled = turn.status === "cancel_requested";
            cancel.innerHTML = '<i class="ri-stop-circle-line" aria-hidden="true"></i><span>取消研究</span>';
            message.append(cancel);
        }
        appendEvidenceIndex(turn, message);
        timeline.append(message);
    }

    function renderConversation() {
        const timeline = $("research-chat-timeline");
        timeline.replaceChildren();
        if (!state.turns.length) {
            const empty = document.createElement("section");
            const copy = document.createElement("p");
            const actions = document.createElement("div");
            empty.className = "research-chat-empty";
            copy.textContent = "从一个问题开始，或把资料带进对话。";
            actions.className = "research-empty-actions";
            [
                ["question", "直接提问"],
                ["file", "添加资料"],
                ["url", "导入链接"],
            ].forEach(([action, label]) => {
                const button = document.createElement("button");
                button.type = "button";
                button.dataset.researchEmptyAction = action;
                button.textContent = label;
                actions.append(button);
            });
            empty.append(copy, actions);
            timeline.append(empty);
            return;
        }
        state.turns.forEach(appendChatEntry);
        timeline.scrollTop = timeline.scrollHeight;
    }

    function renderAttachments() {
        const strip = $("research-attachment-strip");
        const selected = state.files.filter((file) => state.selectedFileIds.has(file.id));
        strip.replaceChildren();
        strip.hidden = !selected.length;
        selected.forEach((file) => {
            const chip = document.createElement("div");
            const name = document.createElement("span");
            const status = document.createElement("small");
            const toggle = document.createElement("button");
            chip.className = "research-attachment-chip";
            name.textContent = file.name;
            status.textContent = sourceStatusLabel(file.status);
            toggle.type = "button";
            toggle.dataset.toggleSource = file.id;
            toggle.title = "本次不使用";
            toggle.setAttribute("aria-label", `本次不使用 ${file.name}`);
            toggle.innerHTML = '<i class="ri-close-line" aria-hidden="true"></i>';
            chip.append(name, status, toggle);
            strip.append(chip);
        });
    }

    function renderFiles() {
        const list = $("research-source-list");
        $("research-source-count").textContent = `${state.files.length} / 10`;
        $("research-stage-source-count").textContent = `${state.files.length} 份资料`;
        list.replaceChildren();
        if (!state.files.length) {
            const empty = document.createElement("p");
            empty.className = "research-empty";
            empty.textContent = "尚未添加资料。";
            list.append(empty);
            return;
        }
        state.files.forEach((file) => {
            const item = document.createElement("div");
            const meta = document.createElement("div");
            const name = document.createElement("strong");
            const status = document.createElement("span");
            const actions = document.createElement("div");
            const toggle = document.createElement("button");
            const remove = document.createElement("button");
            item.className = "research-source-item";
            name.textContent = file.name;
            status.textContent = sourceStatusLabel(file.status);
            meta.append(name, status);
            toggle.type = "button";
            toggle.dataset.toggleSource = file.id;
            toggle.className = "research-source-action";
            toggle.textContent = state.selectedFileIds.has(file.id) ? "本次使用" : "暂不使用";
            remove.type = "button";
            remove.dataset.removeSource = file.id;
            remove.className = "research-source-action research-source-remove";
            remove.title = "从本会话移除";
            remove.setAttribute("aria-label", `从本会话移除 ${file.name}`);
            remove.innerHTML = '<i class="ri-delete-bin-6-line" aria-hidden="true"></i>';
            actions.append(toggle, remove);
            item.append(meta, actions);
            list.append(item);
        });
    }

    function renderAll() {
        $("research-active-session-title").textContent = state.activeSessionTitle || "新建研究";
        const archiveState = $("research-local-archive-state");
        const archiveIcon = document.createElement("i");
        archiveIcon.className = "ri-computer-line";
        archiveIcon.setAttribute("aria-hidden", "true");
        archiveState.replaceChildren(archiveIcon, document.createTextNode(state.archiveAvailable ? "本机存档" : "临时会话"));
        renderConversation();
        renderFiles();
        renderAttachments();
        renderArchiveSessions();
    }

    function formatSessionTime(value) {
        const date = new Date(value);
        const now = new Date();
        if (date.toDateString() === now.toDateString()) return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
        return date.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
    }

    function sessionGroup(value) {
        const date = new Date(value);
        const now = new Date();
        const days = Math.floor((now - date) / 86400000);
        if (date.toDateString() === now.toDateString()) return "今天";
        return days < 7 ? "最近 7 天" : "更早";
    }

    function renderArchiveSessions() {
        const list = $("research-session-list");
        if (!list) return;
        const query = $("research-session-search")?.value || "";
        const sessions = query
            ? state.archiveSessions.filter((session) => (
                session.title.toLocaleLowerCase().includes(query.toLocaleLowerCase())
                || session.messages.some((message) => message.content.toLocaleLowerCase().includes(query.toLocaleLowerCase()))
            ))
            : state.archiveSessions;
        list.replaceChildren();
        if (!sessions.length) {
            const empty = document.createElement("p");
            empty.className = "research-session-empty";
            empty.textContent = query ? "未找到本机会话。" : "尚未保存会话。";
            list.append(empty);
            return;
        }
        let previousGroup = "";
        sessions.forEach((session) => {
            const group = sessionGroup(session.updatedAt);
            if (group !== previousGroup) {
                const label = document.createElement("p");
                label.className = "research-session-group";
                label.textContent = group;
                list.append(label);
                previousGroup = group;
            }
            const row = document.createElement("div");
            const open = document.createElement("button");
            const rename = document.createElement("button");
            const remove = document.createElement("button");
            row.className = `research-session-row${session.id === state.activeSessionId ? " is-active" : ""}`;
            open.type = "button";
            open.dataset.sessionSwitch = session.id;
            open.innerHTML = '<span class="research-signal-cut" aria-hidden="true"></span>';
            const title = document.createElement("strong");
            const meta = document.createElement("small");
            title.textContent = session.title;
            meta.textContent = `${formatSessionTime(session.updatedAt)} · ${session.sources.length} 份资料`;
            open.append(title, meta);
            rename.type = "button";
            rename.className = "research-session-row-action";
            rename.dataset.sessionRename = session.id;
            rename.setAttribute("aria-label", `重命名 ${session.title}`);
            rename.innerHTML = '<i class="ri-pencil-line" aria-hidden="true"></i>';
            remove.type = "button";
            remove.className = "research-session-row-action research-session-row-delete";
            remove.dataset.sessionDelete = session.id;
            remove.setAttribute("aria-label", `删除 ${session.title}`);
            remove.innerHTML = '<i class="ri-delete-bin-6-line" aria-hidden="true"></i>';
            row.append(open, rename, remove);
            list.append(row);
        });
    }

    function toggleSource(fileId) {
        const file = state.files.find((item) => item.id === fileId);
        if (!file) return;
        if (file.status === "expired") return toast("资料已过期，无法用于新的研究请求");
        if (state.selectedFileIds.has(fileId)) state.selectedFileIds.delete(fileId);
        else state.selectedFileIds.add(fileId);
        persistConversation();
        renderAll();
    }

    function removeSource(fileId) {
        state.files = state.files.filter((file) => file.id !== fileId);
        state.selectedFileIds.delete(fileId);
        state.turns.forEach((turn) => {
            if (turn.route === "research" && !turn.jobId && turn.fileIds.includes(fileId) && !TERMINAL_JOB_STATES.has(turn.status)) {
                turn.status = "failed";
                turn.error = "资料处理失败，请移除后重新发送";
            }
        });
        persistConversation();
        renderAll();
    }

    function openSessionLimitDialog() {
        $("research-session-limit-dialog").showModal();
    }

    function openRenameSessionDialog(sessionId) {
        const session = state.archiveSessions.find((item) => item.id === sessionId);
        if (!session) return;
        state.pendingRenameSessionId = sessionId;
        $("research-session-rename-input").value = session.title;
        $("research-session-rename-dialog").showModal();
        $("research-session-rename-input").focus();
        $("research-session-rename-input").select();
    }

    async function confirmArchiveRename() {
        const sessionId = state.pendingRenameSessionId;
        const title = $("research-session-rename-input").value.trim().slice(0, 48);
        $("research-session-rename-dialog").close();
        state.pendingRenameSessionId = "";
        if (!title || !state.archiveAvailable) return;
        const session = state.archiveSessions.find((item) => item.id === sessionId);
        if (!session) return;
        const result = await state.archive.save({ ...session, title });
        if (!result.ok) return toast("本机记录暂时无法保存");
        if (sessionId === state.activeSessionId) state.activeSessionTitle = title;
        await refreshArchiveSessions();
        renderAll();
    }

    function openDeleteSessionDialog(sessionId, mode = "delete") {
        state.pendingDeleteSessionId = sessionId;
        state.pendingDeleteMode = mode;
        const copy = $("research-session-delete-copy");
        if (mode === "clear") copy.textContent = "这会删除此设备中的全部会话记录，服务器资料不会受到影响。";
        else {
            const session = state.archiveSessions.find((item) => item.id === sessionId);
            copy.textContent = `将从此设备删除“${session?.title || "该会话"}”。服务器资料不会受到影响。`;
        }
        $("research-session-delete-dialog").showModal();
    }

    async function confirmArchiveDeletion() {
        const dialog = $("research-session-delete-dialog");
        const mode = state.pendingDeleteMode;
        const sessionId = state.pendingDeleteSessionId;
        dialog.close();
        state.pendingDeleteMode = "";
        state.pendingDeleteSessionId = "";
        if (!state.archiveAvailable) return;
        if (mode === "clear") {
            await state.archive.clear();
            state.archiveSessions = [];
            state.activeSessionId = "";
            await createArchiveSession();
            return;
        }
        await state.archive.remove(sessionId);
        const sessions = await state.archive.list();
        if (sessionId === state.activeSessionId) {
            const next = sessions[0];
            if (next) await switchArchiveSession(next.id);
            else {
                state.activeSessionId = "";
                await createArchiveSession();
            }
        } else {
            state.archiveSessions = sessions;
            renderArchiveSessions();
        }
    }

    async function clearSourceSessionState() {
        clearTimeout(state.fileTimer);
        state.jobTimers.forEach((timer) => clearTimeout(timer));
        state.jobTimers.clear();
        state.files = [];
        state.selectedFileIds.clear();
        state.activeJobIds.clear();
        state.turns.forEach((turn) => {
            if (turn.route === "research" && !TERMINAL_JOB_STATES.has(turn.status)) {
                turn.status = "failed";
                turn.error = "资料会话已清除，请重新发送";
                turn.jobId = null;
            }
        });
        if (state.archiveAvailable) await MergeneticResearchArchive.invalidateSources();
        persistConversation();
        renderAll();
    }

    async function rehydrateFiles() {
        if (!state.key || !state.files.length) return;
        const current = [...state.files];
        const results = await Promise.all(current.map(async (file) => {
            try {
                const data = await api(`/api/model-gateway/files/${encodeURIComponent(file.id)}`, { headers: headers() });
                return { file: normalizeFile(data.file) };
            } catch (error) {
                return { file, error };
            }
        }));
        state.files = results.map((result) => {
            if (result.error?.status === 404) return { ...result.file, status: "expired" };
            return result.file;
        });
        const knownIds = new Set(state.files.map((file) => file.id));
        state.selectedFileIds = new Set([...state.selectedFileIds].filter((id) => (
            knownIds.has(id) && state.files.find((file) => file.id === id)?.status !== "expired"
        )));
        results.filter((result) => result.error && result.error.status !== 404).forEach((result) => toast(`资料状态读取失败：${result.error.message}`));
        persistConversation();
        renderAll();
    }

    async function pollFiles() {
        clearTimeout(state.fileTimer);
        if (!state.files.length || !state.key) return;
        await rehydrateFiles();
        await resumePendingTurns();
        if (state.files.some((file) => PENDING_FILE_STATES.has(file.status))) {
            state.fileTimer = setTimeout(() => { pollFiles().catch((error) => toast(error.message)); }, POLL_MS);
        }
    }

    async function refreshModels(silent) {
        const data = await api("/v1/models", { headers: headers() });
        const select = $("research-model");
        select.replaceChildren();
        (data.data || []).forEach((model) => {
            const option = document.createElement("option");
            option.value = model.id;
            option.textContent = model.id;
            select.append(option);
        });
        if (!data.data?.length) {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = "当前没有可用模型";
            select.append(option);
        }
        if ([...select.options].some((option) => option.value === state.selectedModel)) select.value = state.selectedModel;
        else state.selectedModel = select.value || "";
        persistConversation();
        if (!silent) toast("模型连接已更新");
    }

    async function loadModels(silent) {
        const nextKey = $("research-api-key").value.trim();
        if (!nextKey) throw new Error("请输入 API Key");
        if (state.key && state.key !== nextKey) await clearSourceSessionState();
        state.key = nextKey;
        persistKey();
        await refreshModels(silent);
        await rehydrateFiles();
        await resumePendingTurns();
    }

    async function addUpload(file) {
        const form = new FormData();
        form.append("file", file);
        const data = await api("/api/model-gateway/files", { method: "POST", headers: headers(), body: form });
        const source = normalizeFile(data.file);
        state.files.push(source);
        state.selectedFileIds.add(source.id);
        persistConversation();
        renderAll();
        setToolsOpen(false);
        pollFiles().catch((error) => toast(error.message));
        toast("资料已进入安全处理队列");
    }

    async function addUrl(url) {
        const data = await api("/api/model-gateway/sources/url", {
            method: "POST",
            headers: headers(true),
            body: JSON.stringify({ url }),
        });
        const source = normalizeFile(data.file);
        state.files.push(source);
        state.selectedFileIds.add(source.id);
        $("research-source-url").value = "";
        persistConversation();
        renderAll();
        setToolsOpen(false);
        pollFiles().catch((error) => toast(error.message));
        toast("链接已进入下载队列");
    }

    function contextMessages() {
        return state.turns
            .filter((turn) => (
                (turn.kind === "user" && ["submitting", "running", "completed"].includes(turn.status))
                || (turn.kind === "assistant" && turn.status === "completed")
            ))
            .slice(-MAX_MODEL_CONTEXT_MESSAGES)
            .map((turn) => ({ role: turn.kind === "user" ? "user" : "assistant", content: turn.content }));
    }

    function addAssistantTurn(userTurn, content, citations = [], sources = []) {
        if (state.turns.some((turn) => turn.replyTo === userTurn.id)) return;
        state.turns.push({
            id: newId(),
            kind: "assistant",
            content,
            route: userTurn.route,
            status: "completed",
            fileIds: [],
            model: userTurn.model,
            jobId: null,
            idempotencyKey: null,
            citations,
            sources,
            replyTo: userTurn.id,
            error: "",
        });
    }

    async function submitDirectChat(turn) {
        turn.status = "running";
        persistConversation();
        renderConversation();
        try {
            const data = await api("/v1/chat/completions", {
                method: "POST",
                headers: headers(true),
                body: JSON.stringify({
                    model: turn.model,
                    messages: contextMessages(),
                    temperature: 0.2,
                    max_tokens: 1024,
                    stream: false,
                }),
            });
            const answer = data.choices?.[0]?.message?.content?.trim();
            if (!answer) throw new Error("模型未在同步等待时间内返回可显示的回答");
            turn.status = "completed";
            addAssistantTurn(turn, answer);
        } catch (error) {
            turn.status = "failed";
            turn.error = error.message;
        }
        persistConversation();
        renderConversation();
    }

    function applyResearchJob(turn, job) {
        turn.jobId = job.id || turn.jobId;
        turn.status = job.status || turn.status;
        if (turn.jobId && !TERMINAL_JOB_STATES.has(turn.status)) state.activeJobIds.add(turn.jobId);
        if (job.status === "completed") {
            turn.status = "completed";
            const answer = job.result?.answer || "任务已完成，但结果已不可用。";
            addAssistantTurn(turn, answer, job.result?.citations || [], job.result?.sources || []);
        } else if (TERMINAL_JOB_STATES.has(job.status)) {
            turn.status = job.status;
            turn.error = job.error_code || (job.status === "canceled" ? "研究已取消" : "任务未完成");
            if (turn.jobId) state.activeJobIds.delete(turn.jobId);
        }
    }

    async function submitSelectedSources(turn) {
        if (turn.jobId || turn.status === "submitting") return;
        turn.idempotencyKey = turn.idempotencyKey || newId();
        turn.status = "submitting";
        persistConversation();
        renderConversation();
        try {
            const data = await api("/api/model-gateway/research/jobs", {
                method: "POST",
                headers: Object.assign(headers(true), { "Idempotency-Key": turn.idempotencyKey }),
                body: JSON.stringify({
                    model: turn.model,
                    task_type: "document_qa",
                    file_ids: turn.fileIds,
                    input: turn.content,
                    output_format: "markdown",
                    require_citations: true,
                }),
            });
            applyResearchJob(turn, data.job);
            persistConversation();
            renderConversation();
            await pollResearchTurn(turn.id);
        } catch (error) {
            turn.status = "failed";
            turn.error = error.status === 409 ? "资料处理失败，请移除后重新发送" : error.message;
            persistConversation();
            renderConversation();
        }
    }

    async function pollResearchTurn(turnId) {
        const turn = findTurn(turnId);
        if (!turn?.jobId || !state.key) return;
        clearTimeout(state.jobTimers.get(turnId));
        try {
            const data = await api(`/api/model-gateway/research/jobs/${encodeURIComponent(turn.jobId)}`, { headers: headers() });
            applyResearchJob(turn, data.job);
            persistConversation();
            renderConversation();
            if (!TERMINAL_JOB_STATES.has(turn.status)) {
                state.jobTimers.set(turnId, setTimeout(() => { pollResearchTurn(turnId).catch((error) => toast(error.message)); }, POLL_MS));
            }
        } catch (error) {
            turn.status = "failed";
            turn.error = `无法读取任务状态：${error.message}`;
            persistConversation();
            renderConversation();
        }
    }

    async function cancelResearchTurn(turnId) {
        const turn = findTurn(turnId);
        if (!turn?.jobId) throw new Error("当前研究任务不可取消");
        const data = await api(`/api/model-gateway/research/jobs/${encodeURIComponent(turn.jobId)}/cancel`, {
            method: "POST",
            headers: headers(),
        });
        applyResearchJob(turn, data.job);
        persistConversation();
        renderConversation();
        if (!TERMINAL_JOB_STATES.has(turn.status)) await pollResearchTurn(turn.id);
        toast(turn.status === "canceled" ? "研究已取消" : "已请求取消，正在等待执行器停止");
    }

    async function resumePendingTurns() {
        for (const turn of state.turns.filter((item) => item.route === "research" && !TERMINAL_JOB_STATES.has(item.status))) {
            if (turn.jobId) {
                pollResearchTurn(turn.id).catch((error) => toast(error.message));
                continue;
            }
            const readiness = sourceState(turn.fileIds);
            if (readiness === "failed") {
                turn.status = "failed";
                turn.error = turn.fileIds.some((id) => state.files.find((file) => file.id === id)?.status === "expired")
                    ? "资料已过期，无法用于新的研究请求"
                    : "资料处理失败，请移除后重新发送";
                continue;
            }
            if (readiness === "ready" && turn.status === "preparing") await submitSelectedSources(turn);
        }
        persistConversation();
        renderConversation();
    }

    async function createTurnFromComposer() {
        const content = $("research-question").value.trim();
        const model = $("research-model").value;
        if (!content) throw new Error("请输入问题");
        if (!state.key || !model) throw new Error("请先在会话设置中连接一个正在运行的模型");
        const fileIds = selectedSourceIds();
        const turn = {
            id: newId(),
            kind: "user",
            content,
            route: fileIds.length ? "research" : "direct",
            status: fileIds.length ? "preparing" : "submitting",
            fileIds,
            model,
            jobId: null,
            idempotencyKey: null,
            citations: [],
            sources: [],
            replyTo: null,
            error: "",
        };
        state.turns.push(turn);
        if (state.activeSessionTitle === "新建研究") state.activeSessionTitle = sessionTitleFromTurns(state.turns);
        $("research-question").value = "";
        persistConversation();
        renderAll();
        if (turn.route === "direct") await submitDirectChat(turn);
        else await resumePendingTurns();
    }

    function setToolsOpen(open) {
        $("research-add-source-menu").hidden = !open;
        $("research-toggle-tools").setAttribute("aria-expanded", String(open));
        if (!open) $("research-url-panel").hidden = true;
    }

    function setSettingsOpen(open, trigger) {
        const drawer = $("research-settings-drawer");
        const overlay = $("research-settings-overlay");
        if (open) state.lastSettingsTrigger = trigger || document.activeElement;
        drawer.hidden = !open;
        overlay.hidden = !open;
        drawer.setAttribute("aria-hidden", String(!open));
        $("research-session-settings").setAttribute("aria-expanded", String(open));
        if (open) $("research-close-settings").focus();
        else state.lastSettingsTrigger?.focus();
    }

    function bindEvents() {
        $("research-session-settings").addEventListener("click", (event) => setSettingsOpen(true, event.currentTarget));
        $("research-close-settings").addEventListener("click", () => setSettingsOpen(false));
        $("research-settings-overlay").addEventListener("click", () => setSettingsOpen(false));
        $("research-toggle-tools").addEventListener("click", () => setToolsOpen($("research-add-source-menu").hidden));
        $("research-open-url-panel").addEventListener("click", () => {
            $("research-url-panel").hidden = false;
            $("research-source-url").focus();
        });
        $("research-open-materials").addEventListener("click", () => {
            setToolsOpen(false);
            setSettingsOpen(true, $("research-toggle-tools"));
            $("research-source-list").focus?.();
        });
        $("research-load-models").addEventListener("click", () => loadModels(false).catch((error) => toast(error.message)));
        $("research-model").addEventListener("change", () => {
            state.selectedModel = $("research-model").value;
            persistConversation();
        });
        $("research-clear-key").addEventListener("click", async () => {
            await clearSourceSessionState();
            state.key = "";
            $("research-api-key").value = "";
            $("research-model").replaceChildren(new Option("连接 API Key 后选择", ""));
            state.selectedModel = "";
            localStorage.removeItem("mergeneticResearchKey");
            sessionStorage.removeItem("mergeneticResearchKey");
            persistConversation();
            toast("API Key 已清除");
        });
        $("research-file-input").addEventListener("change", (event) => {
            const file = event.target.files[0];
            if (file) addUpload(file).catch((error) => toast(error.message));
            event.target.value = "";
        });
        $("research-submit-url").addEventListener("click", () => {
            addUrl($("research-source-url").value.trim()).catch((error) => toast(error.message));
        });
        $("research-job-form").addEventListener("submit", (event) => {
            event.preventDefault();
            createTurnFromComposer().catch((error) => toast(error.message));
        });
        $("research-chat-timeline").addEventListener("click", (event) => {
            const action = event.target.closest("[data-research-empty-action]")?.dataset.researchEmptyAction;
            if (action === "question") $("research-question").focus();
            else if (action === "file") $("research-file-input").click();
            else if (action === "url") {
                setToolsOpen(true);
                $("research-url-panel").hidden = false;
                $("research-source-url").focus();
            }
            if (action) return;
            const button = event.target.closest("[data-cancel-turn]");
            if (button) cancelResearchTurn(button.dataset.cancelTurn).catch((error) => toast(error.message));
        });
        const sourceAction = (event) => {
            const remove = event.target.closest("[data-remove-source]");
            const toggle = event.target.closest("[data-toggle-source]");
            if (remove) removeSource(remove.dataset.removeSource);
            else if (toggle) toggleSource(toggle.dataset.toggleSource);
        };
        $("research-attachment-strip").addEventListener("click", sourceAction);
        $("research-source-list").addEventListener("click", sourceAction);
        $("research-new-session").addEventListener("click", () => createArchiveSession().catch((error) => toast(error.message)));
        $("research-session-search").addEventListener("input", renderArchiveSessions);
        $("research-session-list").addEventListener("click", (event) => {
            const switchButton = event.target.closest("[data-session-switch]");
            const renameButton = event.target.closest("[data-session-rename]");
            const deleteButton = event.target.closest("[data-session-delete]");
            if (switchButton) switchArchiveSession(switchButton.dataset.sessionSwitch).catch((error) => toast(error.message));
            else if (renameButton) openRenameSessionDialog(renameButton.dataset.sessionRename);
            else if (deleteButton) openDeleteSessionDialog(deleteButton.dataset.sessionDelete);
        });
        $("research-clear-local-history").addEventListener("click", () => openDeleteSessionDialog("", "clear"));
        $("research-session-delete-cancel").addEventListener("click", () => $("research-session-delete-dialog").close());
        $("research-session-delete-confirm").addEventListener("click", () => confirmArchiveDeletion().catch((error) => toast(error.message)));
        $("research-session-rename-cancel").addEventListener("click", () => $("research-session-rename-dialog").close());
        $("research-session-rename-confirm").addEventListener("click", () => confirmArchiveRename().catch((error) => toast(error.message)));
        $("research-session-limit-cancel").addEventListener("click", () => $("research-session-limit-dialog").close());
        $("research-session-limit-manage").addEventListener("click", () => {
            $("research-session-limit-dialog").close();
            $("research-session-search").focus();
        });
        $("research-session-drawer-toggle").addEventListener("click", (event) => setSessionRailOpen(true, event.currentTarget));
        document.addEventListener("keydown", (event) => {
            if (event.key !== "Escape") return;
            if (!$("research-add-source-menu").hidden) setToolsOpen(false);
            else if (!$("research-settings-drawer").hidden) setSettingsOpen(false);
            else if ($("research-session-rail").classList.contains("is-open")) setSessionRailOpen(false);
        });
    }

    function setSessionRailOpen(open, trigger) {
        const rail = $("research-session-rail");
        if (open) state.lastSessionRailTrigger = trigger || document.activeElement;
        rail.classList.toggle("is-open", open);
        $("research-session-drawer-toggle").setAttribute("aria-expanded", String(open));
        if (open) $("research-session-search").focus();
        else state.lastSessionRailTrigger?.focus();
    }

    document.addEventListener("DOMContentLoaded", async () => {
        $("research-api-key").value = state.key;
        $("research-remember-key").checked = Boolean(localStorage.getItem("mergeneticResearchKey"));
        bindEvents();
        await initializeArchive();
        renderAll();
        runArchiveEntranceMotion();
        const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
        reducedMotion.addEventListener?.("change", (event) => { if (event.matches) killResearchMotion(); });
        window.addEventListener("pagehide", killResearchMotion, { once: true });
        if (state.key) {
            try {
                await loadModels(true);
                await pollFiles();
                await resumePendingTurns();
            } catch (error) {
                toast(`会话恢复需要重新连接模型：${error.message}`);
            }
        }
    });
}());
