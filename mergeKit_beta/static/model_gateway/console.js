(function () {
    const gatewayState = {
        adminToken: "",
        services: [],
        apiKeys: [],
        userModels: [],
        publishableModels: []
    };

    const $ = (id) => document.getElementById(id);

    function showToast(message, type) {
        const toast = $("gateway-toast");
        if (!toast) return;
        toast.textContent = message;
        toast.className = `gateway-toast show ${type || ""}`.trim();
        window.clearTimeout(showToast.timer);
        showToast.timer = window.setTimeout(() => {
            toast.className = "gateway-toast";
        }, 3200);
    }

    async function requestJson(url, options) {
        const res = await fetch(url, options || {});
        const text = await res.text();
        let data = {};
        if (text) {
            try {
                data = JSON.parse(text);
            } catch (err) {
                data = { error: { message: text } };
            }
        }
        if (!res.ok) {
            const detail = data.error && data.error.message ? data.error.message : `HTTP ${res.status}`;
            const error = new Error(detail);
            error.status = res.status;
            error.code = data.error && data.error.code;
            throw error;
        }
        return data;
    }

    function adminHeaders() {
        return {
            "Authorization": `Bearer ${gatewayState.adminToken}`,
            "Content-Type": "application/json"
        };
    }

    function setRuntimeSummary() {
        const running = gatewayState.services.filter((svc) => svc.status === "running").length;
        const statRunning = $("gateway-stat-running");
        const statKeys = $("gateway-stat-keys");
        const statModels = $("gateway-stat-models");
        const text = $("gateway-runtime-text");
        const dot = document.querySelector(".gateway-live-dot");
        if (statRunning) statRunning.textContent = String(running);
        if (statKeys) statKeys.textContent = String(gatewayState.apiKeys.length);
        if (statModels) statModels.textContent = String(gatewayState.userModels.length || running);
        if (text) text.textContent = running > 0 ? "模型运行中" : "手动管理";
        if (dot) dot.style.background = running > 0 ? "var(--gateway-success)" : "var(--gateway-warning)";
    }

    function renderServices() {
        const list = $("gateway-services-list");
        if (!list) return;
        if (!gatewayState.adminToken) {
            list.innerHTML = '<div class="gateway-empty">输入 Admin Token 后查看模型服务。</div>';
            setRuntimeSummary();
            return;
        }
        if (!gatewayState.services.length) {
            list.innerHTML = '<div class="gateway-empty">暂无服务。创建服务后，管理员可手动启动。</div>';
            setRuntimeSummary();
            return;
        }
        list.innerHTML = gatewayState.services.map((svc) => {
            const status = svc.status || "unknown";
            const gpuText = (svc.gpu_ids || []).join(",") || "未设置";
            const action = status === "running"
                ? `<button class="gateway-small-btn danger" data-stop-service="${svc.id}">停止</button>`
                : (["stopped", "failed"].includes(status)
                    ? `<button class="gateway-small-btn primary" data-start-service="${svc.id}">启动</button><button class="gateway-small-btn danger" data-delete-service="${svc.id}">删除</button>`
                    : "");
            return `
                <article class="gateway-list-item">
                    <div class="gateway-item-top">
                        <div>
                            <div class="gateway-item-title">${escapeHtml(svc.display_name || svc.served_model_name)}</div>
                            <div class="gateway-item-meta">${escapeHtml(svc.served_model_name || "")}</div>
                        </div>
                        <span class="gateway-status ${escapeHtml(status)}">${escapeHtml(status)}</span>
                    </div>
                    <div class="gateway-item-meta">GPU ${escapeHtml(gpuText)} · TP ${svc.tensor_parallel_size || 1} · ${escapeHtml(svc.dtype || "auto")}</div>
                    ${svc.last_error ? `<div class="gateway-item-meta">错误：${escapeHtml(svc.last_error)}</div>` : ""}
                    <div class="gateway-item-actions">${action}</div>
                </article>
            `;
        }).join("");
        setRuntimeSummary();
    }

    function renderKeys() {
        const list = $("gateway-api-keys-list");
        if (!list) return;
        if (!gatewayState.adminToken) {
            list.innerHTML = '<div class="gateway-empty">输入 Admin Token 后查看 API Key。</div>';
            return;
        }
        if (!gatewayState.apiKeys.length) {
            list.innerHTML = '<div class="gateway-empty">暂无 API Key。创建后明文只显示一次。</div>';
            return;
        }
        list.innerHTML = gatewayState.apiKeys.map((key) => {
            const allowlist = (key.model_allowlist || []).join(", ") || "全部模型";
            return `
                <article class="gateway-list-item">
                    <div class="gateway-item-top">
                        <div>
                            <div class="gateway-item-title">${escapeHtml(key.owner_label || "unnamed")}</div>
                            <div class="gateway-item-meta">${escapeHtml(key.prefix || "mk_live")}...${escapeHtml(key.last4 || "")}</div>
                        </div>
                        <span class="gateway-status ${escapeHtml(key.status || "")}">${escapeHtml(key.status || "unknown")}</span>
                    </div>
                    <div class="gateway-item-meta">白名单：${escapeHtml(allowlist)}</div>
                </article>
            `;
        }).join("");
    }

    function renderUserModels() {
        const select = $("gateway-chat-model");
        if (!select) return;
        if (!gatewayState.userModels.length) {
            select.innerHTML = '<option value="">暂无可调用模型</option>';
            setRuntimeSummary();
            return;
        }
        select.innerHTML = gatewayState.userModels.map((model) => `<option value="${escapeAttr(model.id)}">${escapeHtml(model.id)}</option>`).join("");
        setRuntimeSummary();
    }

    async function loadAdminData() {
        if (!gatewayState.adminToken) {
            renderServices();
            renderKeys();
            return;
        }
        renderPublishedModelsLoading();
        try {
            const [services, keys, models] = await Promise.all([
                requestJson("/api/model-gateway/admin/model-services", { headers: adminHeaders() }),
                requestJson("/api/model-gateway/admin/api-keys", { headers: adminHeaders() }),
                requestJson("/api/model-gateway/admin/publishable-models", { headers: adminHeaders() })
            ]);
            gatewayState.services = services.services || [];
            gatewayState.apiKeys = keys.api_keys || [];
            gatewayState.publishableModels = models.models || [];
            renderServices();
            renderKeys();
            renderPublishedModels();
        } catch (error) {
            gatewayState.publishableModels = [];
            renderPublishedModelsError(error);
            throw error;
        }
    }

    function renderPublishedModelsLoading() {
        const select = $("gateway-published-model");
        const summary = $("gateway-published-summary");
        const unavailable = $("gateway-unavailable-models");
        if (!select || !summary || !unavailable) return;
        select.disabled = true;
        select.innerHTML = '<option value="">加载正式资产...</option>';
        summary.innerHTML = '<span>类型：加载中</span><span>兼容性：加载中</span>';
        unavailable.innerHTML = "";
    }

    function renderPublishedModelsError(error) {
        const select = $("gateway-published-model");
        const summary = $("gateway-published-summary");
        const unavailable = $("gateway-unavailable-models");
        if (!select || !summary || !unavailable) return;
        select.disabled = true;
        select.innerHTML = '<option value="">无法加载正式资产</option>';
        summary.innerHTML = `<span>类型：不可用</span><span>兼容性：${error.status === 401 ? "未授权" : "加载失败"}</span>`;
        unavailable.innerHTML = `<button class="gateway-small-btn" type="button" data-retry-published-models>重试加载正式资产</button>`;
    }

    function renderPublishedModels() {
        const select = $("gateway-published-model");
        const summary = $("gateway-published-summary");
        const unavailable = $("gateway-unavailable-models");
        if (!select || !summary || !unavailable) return;
        if (!gatewayState.adminToken) {
            select.disabled = true;
            select.innerHTML = '<option value="">连接 Admin Token 后加载</option>';
            summary.innerHTML = '<span>类型：待选择</span><span>兼容性：待检查</span>';
            unavailable.innerHTML = "";
            return;
        }
        const selectable = gatewayState.publishableModels.filter((model) => model.selectable);
        const blocked = gatewayState.publishableModels.filter((model) => !model.selectable);
        select.disabled = !selectable.length;
        select.innerHTML = selectable.length
            ? `<option value="">选择正式资产</option>${selectable.map((model) => `<option value="${escapeAttr(model.model_id)}">${escapeHtml(model.display_name || model.model_id)} (${escapeHtml(model.artifact_type || "text")})</option>`).join("")}`
            : '<option value="">暂无可创建服务的正式资产</option>';
        unavailable.innerHTML = blocked.length
            ? `<p class="gateway-unavailable-title">不可创建服务的正式资产</p>${blocked.map((model) => `<div class="gateway-unavailable-item" aria-disabled="true"><strong>${escapeHtml(model.display_name || model.model_id)}</strong><span>${escapeHtml(model.artifact_type || "text")} · ${escapeHtml(model.blocked_reason_code || "blocked")}</span></div>`).join("")}`
            : "";
        updatePublishedModelSummary();
    }

    function updatePublishedModelSummary() {
        const summary = $("gateway-published-summary");
        const select = $("gateway-published-model");
        if (!summary || !select) return;
        const model = gatewayState.publishableModels.find((item) => item.model_id === select.value);
        summary.innerHTML = model
            ? `<span>类型：${escapeHtml(model.artifact_type || "text")}</span><span>兼容性：${model.selectable ? "ready" : escapeHtml(model.blocked_reason_code || "blocked")}</span>`
            : '<span>类型：待选择</span><span>兼容性：待检查</span>';
    }

    async function loadUserModels() {
        const key = ($("gateway-user-api-key") || {}).value || "";
        if (!key.trim()) {
            gatewayState.userModels = [];
            renderUserModels();
            return;
        }
        const data = await requestJson("/v1/models", {
            headers: { "Authorization": `Bearer ${key.trim()}` }
        });
        gatewayState.userModels = data.data || [];
        renderUserModels();
    }

    function bindEvents() {
        const menuBtn = $("gateway-menu-btn");
        const closeBtn = $("gateway-close-sidebar");
        const sidebar = $("gateway-sidebar");
        const overlay = $("gateway-sidebar-overlay");
        const closeSidebar = () => {
            if (sidebar) sidebar.classList.remove("open");
            if (overlay) overlay.classList.remove("show");
        };
        if (menuBtn) menuBtn.addEventListener("click", () => {
            if (sidebar) sidebar.classList.add("open");
            if (overlay) overlay.classList.add("show");
        });
        if (closeBtn) closeBtn.addEventListener("click", closeSidebar);
        if (overlay) overlay.addEventListener("click", closeSidebar);

        const adminForm = $("gateway-admin-token-form");
        if (adminForm) adminForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            gatewayState.adminToken = ($("gateway-admin-token").value || "").trim();
            try {
                await loadAdminData();
                showToast("管理员连接成功", "success");
            } catch (err) {
                showToast(`管理员连接失败：${err.message}`, "error");
            }
        });

        const publishedModel = $("gateway-published-model");
        if (publishedModel) publishedModel.addEventListener("change", updatePublishedModelSummary);
        const unavailableModels = $("gateway-unavailable-models");
        if (unavailableModels) unavailableModels.addEventListener("click", async (event) => {
            if (!event.target.closest("[data-retry-published-models]")) return;
            try {
                await loadAdminData();
                showToast("正式资产已刷新", "success");
            } catch (err) {
                showToast(`正式资产加载失败：${err.message}`, "error");
            }
        });

        const refreshServices = $("gateway-refresh-services");
        if (refreshServices) refreshServices.addEventListener("click", async () => {
            try {
                await loadAdminData();
                showToast("服务列表已刷新", "success");
            } catch (err) {
                showToast(`刷新失败：${err.message}`, "error");
            }
        });

        const serviceForm = $("gateway-create-service-form");
        if (serviceForm) serviceForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            if (!gatewayState.adminToken) {
                showToast("请先连接 Admin Token", "error");
                return;
            }
            const gpuIds = ($("gateway-gpu-ids").value || "")
                .split(",")
                .map((item) => item.trim())
                .filter(Boolean)
                .map((item) => Number(item));
            const payload = {
                model_id: $("gateway-published-model").value,
                display_name: $("gateway-display-name").value.trim(),
                served_model_name: $("gateway-served-name").value.trim(),
                gpu_ids: gpuIds,
                tensor_parallel_size: Number($("gateway-tp").value || 1),
                gpu_memory_utilization: Number($("gateway-gpu-memory").value || 0.85),
                dtype: $("gateway-dtype").value,
                max_num_seqs: Number($("gateway-max-seqs").value || 8),
                trust_remote_code: $("gateway-trust-remote-code").checked
            };
            try {
                await requestJson("/api/model-gateway/admin/model-services", {
                    method: "POST",
                    headers: adminHeaders(),
                    body: JSON.stringify(payload)
                });
                serviceForm.reset();
                renderPublishedModels();
                $("gateway-gpu-ids").value = "0";
                $("gateway-tp").value = "1";
                $("gateway-gpu-memory").value = "0.85";
                $("gateway-max-seqs").value = "8";
                await loadAdminData();
                showToast("服务配置已创建", "success");
            } catch (err) {
                showToast(`创建失败：${err.message}`, "error");
            }
        });

        const servicesList = $("gateway-services-list");
        if (servicesList) servicesList.addEventListener("click", async (event) => {
            const startId = event.target.closest("[data-start-service]")?.getAttribute("data-start-service");
            const stopId = event.target.closest("[data-stop-service]")?.getAttribute("data-stop-service");
            const deleteId = event.target.closest("[data-delete-service]")?.getAttribute("data-delete-service");
            const serviceId = startId || stopId || deleteId;
            if (!serviceId) return;
            const action = startId ? "start" : (stopId ? "stop" : "delete");
            try {
                await requestJson(`/api/model-gateway/admin/model-services/${encodeURIComponent(serviceId)}${action === "delete" ? "" : `/${action}`}`, {
                    method: action === "delete" ? "DELETE" : "POST",
                    headers: adminHeaders()
                });
                await loadAdminData();
                showToast(action === "start" ? "启动请求已提交" : (action === "stop" ? "停止请求已提交" : "服务已删除"), "success");
            } catch (err) {
                const label = action === "start" ? "启动" : (action === "stop" ? "停止" : "删除");
                showToast(`${label}失败：${err.message}`, "error");
            }
        });

        const keyForm = $("gateway-create-key-form");
        if (keyForm) keyForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            if (!gatewayState.adminToken) {
                showToast("请先连接 Admin Token", "error");
                return;
            }
            const allowlist = ($("gateway-key-allowlist").value || "")
                .split(",")
                .map((item) => item.trim())
                .filter(Boolean);
            try {
                const data = await requestJson("/api/model-gateway/admin/api-keys", {
                    method: "POST",
                    headers: adminHeaders(),
                    body: JSON.stringify({
                        owner_label: $("gateway-key-owner").value.trim(),
                        model_allowlist: allowlist
                    })
                });
                const box = $("gateway-new-key");
                if (box) {
                    box.hidden = false;
                    box.textContent = data.api_key;
                }
                keyForm.reset();
                await loadAdminData();
                showToast("API Key 已生成", "success");
            } catch (err) {
                showToast(`生成失败：${err.message}`, "error");
            }
        });

        const refreshModels = $("gateway-refresh-v1-models");
        if (refreshModels) refreshModels.addEventListener("click", async () => {
            try {
                await loadUserModels();
                showToast("可用模型已刷新", "success");
            } catch (err) {
                showToast(`模型刷新失败：${err.message}`, "error");
            }
        });

        const chatForm = $("gateway-chat-form");
        if (chatForm) chatForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            const key = ($("gateway-user-api-key").value || "").trim();
            const model = ($("gateway-chat-model").value || "").trim();
            const output = $("gateway-chat-output");
            const usage = $("gateway-usage-line");
            if (!key || !model) {
                showToast("请填写 API Key 并选择模型", "error");
                return;
            }
            const requestId = window.crypto && window.crypto.randomUUID
                ? window.crypto.randomUUID()
                : "";
            const cancelInput = $("gateway-cancel-request-id");
            if (cancelInput && requestId) cancelInput.value = requestId;
            if (output) output.textContent = "请求中...";
            if (usage) usage.textContent = "usage: pending";
            try {
                const data = await requestJson("/v1/chat/completions", {
                    method: "POST",
                    headers: {
                        "Authorization": `Bearer ${key}`,
                        "Content-Type": "application/json",
                        ...(requestId ? { "X-Request-Id": requestId } : {})
                    },
                    body: JSON.stringify({
                        model,
                        messages: [{ role: "user", content: $("gateway-chat-input").value }],
                        max_tokens: Number($("gateway-max-tokens").value || 512),
                        temperature: Number($("gateway-temperature").value || 0.7),
                        top_p: Number($("gateway-top-p").value || 0.9)
                    })
                });
                if (data.object === "serving.request" && data.request_id) {
                    if (output) {
                        output.textContent = JSON.stringify({
                            request_id: data.request_id,
                            status: data.status,
                            status_url: data.status_url,
                            cancel_url: data.cancel_url
                        }, null, 2);
                    }
                    if (cancelInput) cancelInput.value = data.request_id;
                    if (usage) usage.textContent = "usage: pending";
                    showToast("请求已进入后台，可查询状态", "success");
                    return;
                }
                const message = data.choices && data.choices[0] && data.choices[0].message
                    ? data.choices[0].message.content
                    : JSON.stringify(data, null, 2);
                if (output) output.textContent = message || "";
                if (usage) {
                    const usageData = data.usage || {};
                    usage.textContent = `usage: ${usageData.total_tokens || 0} tokens`;
                }
                showToast("请求完成", "success");
            } catch (err) {
                if (output) output.textContent = err.message;
                if (usage) usage.textContent = "usage: failed";
                showToast(`调用失败：${err.message}`, "error");
            }
        });

        const cancelForm = $("gateway-cancel-request-form");
        if (cancelForm) cancelForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            const key = ($("gateway-user-api-key").value || "").trim();
            const requestId = ($("gateway-cancel-request-id").value || "").trim();
            if (!key || !requestId) {
                showToast("请填写 API Key 和 request_id", "error");
                return;
            }
            try {
                const data = await requestJson(`/v1/requests/${encodeURIComponent(requestId)}/cancel`, {
                    method: "POST",
                    headers: { "Authorization": `Bearer ${key}` }
                });
                const status = data.request && data.request.status ? data.request.status : data.status;
                showToast(`请求状态：${status}`, "success");
            } catch (err) {
                showToast(`取消失败：${err.message}`, "error");
            }
        });

        const checkStatus = $("gateway-check-request-status");
        if (checkStatus) checkStatus.addEventListener("click", async () => {
            const key = ($("gateway-user-api-key").value || "").trim();
            const requestId = ($("gateway-cancel-request-id").value || "").trim();
            const output = $("gateway-request-status-output");
            if (!key || !requestId) {
                showToast("请填写 API Key 和 request_id", "error");
                return;
            }
            if (output) output.textContent = "查询中...";
            try {
                const data = await requestJson(`/v1/requests/${encodeURIComponent(requestId)}`, {
                    headers: { "Authorization": `Bearer ${key}` }
                });
                if (output) {
                    output.textContent = JSON.stringify({
                        request: data.request,
                        usage: data.usage
                    }, null, 2);
                }
                const status = data.request && data.request.status ? data.request.status : "unknown";
                showToast(`请求状态：${status}`, "success");
            } catch (err) {
                if (output) output.textContent = err.message;
                showToast(`查询失败：${err.message}`, "error");
            }
        });

        const copyCurl = $("gateway-copy-curl");
        if (copyCurl) copyCurl.addEventListener("click", async () => {
            const code = copyCurl.parentElement.querySelector("code")?.textContent || "";
            try {
                await navigator.clipboard.writeText(code);
                showToast("示例已复制", "success");
            } catch (err) {
                showToast("当前浏览器不支持复制", "error");
            }
        });
    }

    function escapeHtml(value) {
        return String(value == null ? "" : value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function escapeAttr(value) {
        return escapeHtml(value).replace(/`/g, "&#96;");
    }

    function runEntranceAnimation() {
        if (!window.gsap || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
        const mm = window.gsap.matchMedia();
        mm.add("(min-width: 720px)", () => {
            window.gsap.from(".gateway-hero-copy > *", {
                y: 22,
                autoAlpha: 0,
                duration: 0.72,
                ease: "power3.out",
                stagger: 0.08
            });
            window.gsap.from(".gateway-runtime-board", {
                y: 28,
                scale: 0.98,
                autoAlpha: 0,
                duration: 0.8,
                ease: "power3.out"
            });
            window.gsap.from(".gateway-section, .gateway-panel", {
                y: 18,
                autoAlpha: 0,
                duration: 0.56,
                ease: "power2.out",
                stagger: 0.05,
                delay: 0.16
            });
        });
    }

    document.addEventListener("DOMContentLoaded", async () => {
        bindEvents();
        renderServices();
        renderKeys();
        renderUserModels();
        runEntranceAnimation();
        renderPublishedModels();
    });
})();
