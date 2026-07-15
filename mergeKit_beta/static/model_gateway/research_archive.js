(function () {
    const DB_NAME = "mergeneticResearchArchiveV1";
    const DB_VERSION = 1;
    const STORE_NAME = "sessions";
    const ACTIVE_SESSION_KEY = "mergeneticResearchArchiveActiveSessionId";
    const MAX_SESSIONS = 20;
    const MAX_TURNS = 60;
    const MAX_SESSION_BYTES = 256 * 1024;
    let databasePromise = null;

    function text(value, fallback = "") {
        return typeof value === "string" ? value : fallback;
    }

    function timestamp(value) {
        const parsed = Number(value);
        return Number.isFinite(parsed) && parsed > 0 ? parsed : Date.now();
    }

    function newId() {
        if (window.crypto?.randomUUID) return window.crypto.randomUUID();
        return `research-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function sanitizeLocator(locator) {
        if (!locator || typeof locator !== "object") return null;
        const kind = text(locator.kind);
        const value = Number(locator.value);
        return ["page", "slide", "paragraph"].includes(kind) && Number.isFinite(value) ? { kind, value } : null;
    }

    function sanitizeMessage(message) {
        if (!message || typeof message !== "object") return null;
        const role = ["user", "assistant", "error"].includes(message.role || message.kind) ? (message.role || message.kind) : "error";
        const content = text(message.content).slice(0, 64 * 1024);
        if (!content) return null;
        return {
            id: text(message.id, newId()),
            role,
            content,
            status: text(message.status, "completed"),
            route: message.route === "research" ? "research" : "direct",
            model: text(message.model),
            citations: Array.isArray(message.citations) ? message.citations.map(Number).filter(Number.isInteger).slice(0, 64) : [],
            locators: Array.isArray(message.locators || message.sources)
                ? (message.locators || message.sources).map((item) => sanitizeLocator(item?.locator || item)).filter(Boolean).slice(0, 64)
                : [],
            jobId: text(message.jobId),
            idempotencyKey: text(message.idempotencyKey),
            error: text(message.error).slice(0, 512),
            replyTo: text(message.replyTo),
        };
    }

    function sanitizeSource(source) {
        return {
            id: text(source?.id).slice(0, 256),
            name: text(source?.name, "未命名资料").slice(0, 256),
            sourceKind: text(source?.sourceKind, "upload").slice(0, 32),
            status: text(source?.status, "expired").slice(0, 32),
            selected: Boolean(source?.selected),
        };
    }

    function byteLength(value) {
        return new TextEncoder().encode(JSON.stringify(value)).length;
    }

    function trimToByteLimit(record, limit) {
        while (byteLength(record) > limit && record.messages.length > 1) record.messages.shift();
        if (byteLength(record) > limit && record.messages.length === 1) {
            record.messages[0].content = `${record.messages[0].content.slice(0, 2048)}\n[本机记录已截断]`;
        }
        return record;
    }

    function sanitizeSession(record) {
        const messages = Array.isArray(record?.messages)
            ? record.messages.map(sanitizeMessage).filter(Boolean).slice(-MAX_TURNS)
            : [];
        const sources = Array.isArray(record?.sources) ? record.sources.map(sanitizeSource).slice(0, 10) : [];
        const session = {
            id: text(record?.id),
            title: text(record?.title, "新建研究").slice(0, 48) || "新建研究",
            createdAt: timestamp(record?.createdAt),
            updatedAt: timestamp(record?.updatedAt),
            messages,
            sources,
            selectedModel: text(record?.selectedModel).slice(0, 256),
        };
        if (!session.id) throw new Error("本机会话缺少标识");
        return trimToByteLimit(session, MAX_SESSION_BYTES);
    }

    function open() {
        if (databasePromise) return databasePromise;
        databasePromise = new Promise((resolve, reject) => {
            const request = indexedDB.open(DB_NAME, DB_VERSION);
            request.onerror = () => reject(request.error || new Error("无法打开本机会话记录"));
            request.onupgradeneeded = () => {
                const database = request.result;
                const store = database.objectStoreNames.contains(STORE_NAME)
                    ? request.transaction.objectStore(STORE_NAME)
                    : database.createObjectStore(STORE_NAME, { keyPath: "id" });
                if (!store.indexNames.contains("updatedAt")) store.createIndex("updatedAt", "updatedAt");
            };
            request.onsuccess = () => resolve(request.result);
        });
        return databasePromise;
    }

    async function requestValue(mode, operation) {
        const database = await open();
        return new Promise((resolve, reject) => {
            const transaction = database.transaction(STORE_NAME, mode);
            const store = transaction.objectStore(STORE_NAME);
            const request = operation(store);
            transaction.onerror = () => reject(transaction.error || new Error("本机会话记录操作失败"));
            request.onerror = () => reject(request.error || new Error("本机会话记录操作失败"));
            request.onsuccess = () => resolve(request.result);
        });
    }

    async function list() {
        const records = await requestValue("readonly", (store) => store.getAll());
        return records.map(sanitizeSession).sort((left, right) => right.updatedAt - left.updatedAt);
    }

    async function read(id) {
        if (!id) return null;
        const record = await requestValue("readonly", (store) => store.get(String(id)));
        return record ? sanitizeSession(record) : null;
    }

    async function save(record) {
        const session = sanitizeSession({ ...record, updatedAt: Date.now() });
        const current = await read(session.id);
        if (!current && (await list()).length >= MAX_SESSIONS) return { ok: false, reason: "session_limit" };
        await requestValue("readwrite", (store) => store.put(session));
        return { ok: true, session };
    }

    async function remove(id) {
        if (!id) return;
        await requestValue("readwrite", (store) => store.delete(String(id)));
    }

    async function clear() {
        await requestValue("readwrite", (store) => store.clear());
        localStorage.removeItem(ACTIVE_SESSION_KEY);
    }

    async function search(query) {
        const needle = text(query).trim().toLocaleLowerCase();
        const sessions = await list();
        if (!needle) return sessions;
        return sessions.filter((session) => (
            session.title.toLocaleLowerCase().includes(needle)
            || session.messages.some((message) => message.content.toLocaleLowerCase().includes(needle))
        ));
    }

    function titleFromMessages(messages) {
        const firstQuestion = messages.find((message) => message.role === "user" && message.content)?.content;
        return text(firstQuestion, "新建研究").replace(/\s+/g, " ").trim().slice(0, 48) || "新建研究";
    }

    async function migrateLegacy(legacySnapshot) {
        if (!legacySnapshot || !Array.isArray(legacySnapshot.turns)) return null;
        const messages = legacySnapshot.turns.map((turn) => ({
            ...turn,
            role: turn.kind,
            locators: turn.sources,
        }));
        const record = {
            id: newId(),
            title: titleFromMessages(messages),
            createdAt: Date.now(),
            updatedAt: Date.now(),
            messages,
            sources: (legacySnapshot.files || []).map((file) => ({
                ...file,
                selected: (legacySnapshot.selectedFileIds || []).map(String).includes(String(file.id)),
            })),
            selectedModel: legacySnapshot.selectedModel,
        };
        const result = await save(record);
        return result.ok ? result.session.id : null;
    }

    async function invalidateSources() {
        const sessions = await list();
        const affected = [];
        for (const session of sessions) {
            if (!session.sources.length) continue;
            session.sources = session.sources.map((source) => ({ ...source, id: "", selected: false, status: "expired" }));
            await save(session);
            affected.push(session.id);
        }
        return affected;
    }

    window.MergeneticResearchArchive = {
        open,
        list,
        read,
        save,
        remove,
        clear,
        search,
        migrateLegacy,
        invalidateSources,
    };
}());
