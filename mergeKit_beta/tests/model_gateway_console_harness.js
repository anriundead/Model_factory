"use strict";

const assert = require("assert");
const fs = require("fs");
const vm = require("vm");

class Element {
    constructor(id) {
        this.id = id;
        this.value = "";
        this.checked = false;
        this.disabled = false;
        this.hidden = false;
        this.innerHTML = "";
        this.textContent = "";
        this.listeners = {};
        this.children = [];
    }

    addEventListener(type, callback) {
        this.listeners[type] = callback;
    }

    dispatch(type, event = {}) {
        const callback = this.listeners[type];
        if (callback) return callback({ target: this, preventDefault() {}, ...event });
        return undefined;
    }

    append(...children) {
        this.children.push(...children);
    }

    replaceChildren(...children) {
        this.children = children;
        this.innerHTML = "";
    }

    querySelector() {
        return null;
    }
}

const ids = [
    "gateway-admin-token-form", "gateway-admin-token", "gateway-services-list",
    "gateway-api-keys-list", "gateway-published-model", "gateway-published-summary",
    "gateway-unavailable-models", "gateway-stat-running", "gateway-stat-keys",
    "gateway-stat-models", "gateway-runtime-text", "gateway-refresh-services",
    "gateway-create-service-form", "gateway-display-name", "gateway-served-name",
    "gateway-gpu-ids", "gateway-tp", "gateway-gpu-memory", "gateway-dtype",
    "gateway-max-seqs", "gateway-trust-remote-code", "gateway-user-api-key",
    "gateway-chat-model", "gateway-refresh-v1-models", "gateway-chat-form",
    "gateway-chat-input", "gateway-max-tokens", "gateway-temperature", "gateway-top-p",
    "gateway-cancel-request-id", "gateway-chat-output", "gateway-usage-line",
    "gateway-cancel-request-form", "gateway-check-request-status", "gateway-request-status-output",
    "gateway-copy-curl", "gateway-new-key", "gateway-key-owner", "gateway-key-allowlist",
    "gateway-create-key-form", "gateway-menu-btn", "gateway-close-sidebar",
    "gateway-sidebar", "gateway-sidebar-overlay", "gateway-toast"
];
const elements = Object.fromEntries(ids.map((id) => [id, new Element(id)]));
const document = {
    addEventListener(type, callback) {
        if (type === "DOMContentLoaded") this.domReady = callback;
    },
    createElement(tagName) {
        const element = new Element("");
        element.tagName = tagName.toUpperCase();
        return element;
    },
    getElementById(id) {
        return elements[id] || null;
    },
    querySelector() {
        return null;
    }
};

const candidates = [{
    model_id: "model-x",
    display_name: "Model X",
    artifact_type: "vlm<script>",
    model_path: "/data/PublishedModels/model-x/" + "x".repeat(220),
    selectable: true
}, {
    model_id: "blocked-model",
    display_name: "Blocked Model",
    artifact_type: "vlm",
    model_path: "/data/PublishedModels/blocked-model",
    selectable: false,
    blocked_reason_code: "unsupported_architecture"
}];
const context = {
    console,
    document,
    window: {
        matchMedia: () => ({ matches: true }),
        clearTimeout() {},
        setTimeout() { return 1; },
        crypto: { randomUUID: () => "request-id" }
    },
    fetch: async (url) => ({
        ok: true,
        status: 200,
        async text() {
            if (url.endsWith("/publishable-models")) return JSON.stringify({ models: candidates });
            if (url.endsWith("/model-services")) return JSON.stringify({ services: [] });
            return JSON.stringify({ api_keys: [] });
        }
    })
};
vm.runInNewContext(
    fs.readFileSync(require.resolve("../static/model_gateway/console.js"), "utf8"),
    context,
    { filename: "console.js" }
);

(async () => {
    await document.domReady();
    elements["gateway-admin-token"].value = "admin";
    await elements["gateway-admin-token-form"].dispatch("submit");
    const select = elements["gateway-published-model"];
    select.value = "model-x";
    elements["gateway-published-summary"].dispatch("change");
    await select.dispatch("change");
    const summary = elements["gateway-published-summary"];
    assert.deepStrictEqual(summary.children.map((child) => child.textContent), [
        "类型：vlm<script>",
        "路径：/data/PublishedModels/model-x/" + "x".repeat(220),
        "兼容性：ready"
    ]);
    assert.strictEqual(summary.innerHTML, "", "summary must be built with DOM text nodes");
    assert.strictEqual(summary.children[0].children.length, 0, "candidate text must not create markup");
    select.value = "";
    await select.dispatch("change");
    assert.deepStrictEqual(summary.children.map((child) => child.textContent), [
        "类型：待选择", "路径：待选择", "兼容性：待检查"
    ]);
    select.value = "blocked-model";
    await select.dispatch("change");
    assert.deepStrictEqual(summary.children.map((child) => child.textContent), [
        "类型：待选择", "路径：待选择", "兼容性：待检查"
    ], "blocked stale selection must clear the formal asset summary");
    console.log("gateway-console harness: summary textContent and empty-state assertions passed");
})().catch((error) => {
    console.error(error.stack || error);
    process.exitCode = 1;
});
