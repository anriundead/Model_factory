(function (root, factory) {
    const api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    else root.PublicationUI = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    const lifecycle = {
        queued: { segment: 0, kind: "active" },
        recipe: { segment: 0, kind: "active" },
        materializing: { segment: 1, kind: "active" },
        running: { segment: 1, kind: "active" },
        validating: { segment: 2, kind: "active" },
        registration_pending: { segment: 3, kind: "active" },
        completed: { segment: 3, kind: "completed" },
        failed: { segment: null, kind: "failed" },
        canceled: { segment: null, kind: "canceled" },
        unknown: { segment: null, kind: "unknown" },
        loading: { segment: null, kind: "loading" }
    };

    function lifecycleForStatus(status) {
        const value = lifecycle[String(status || "unknown")] || lifecycle.unknown;
        return { segment: value.segment, kind: value.kind, gateway: false };
    }

    function createGenerationGate() {
        let generation = 0;
        let token = "";
        return {
            changeToken(nextToken) {
                const next = String(nextToken || "");
                if (next === token) return false;
                token = next;
                generation += 1;
                return true;
            },
            snapshot() {
                return { generation, token };
            },
            isCurrent(candidate) {
                return !!candidate && candidate.generation === generation && candidate.token === token;
            }
        };
    }

    function createPollScheduler(options) {
        const setTimer = options.setTimer;
        const clearTimer = options.clearTimer;
        const normalDelay = options.normalDelay;
        const retryBaseDelay = options.retryBaseDelay;
        const retryMaxDelay = options.retryMaxDelay;
        let timer = null;
        let retryDelay = retryBaseDelay;

        function stop() {
            if (timer !== null) clearTimer(timer);
            timer = null;
        }

        return {
            schedule(callback, transient) {
                stop();
                const delay = transient ? retryDelay : normalDelay;
                if (transient) retryDelay = Math.min(retryDelay * 2, retryMaxDelay);
                timer = setTimer(function () {
                    timer = null;
                    callback();
                }, delay);
                return delay;
            },
            succeeded() {
                retryDelay = retryBaseDelay;
            },
            stop,
            hasTimer() {
                return timer !== null;
            }
        };
    }

    function createPollController(options) {
        const poller = options.poller;

        async function poll(ids, refreshTask, snapshot) {
            const results = await Promise.allSettled(ids.map((id) => refreshTask(id)));
            if (options.isCurrent && !options.isCurrent(snapshot)) return { stale: true };

            let successful = false;
            let transientError = null;
            let authError = null;
            results.forEach((result, index) => {
                const id = ids[index];
                if (result.status === "fulfilled") {
                    options.acceptTask(id, result.value);
                    successful = true;
                    return;
                }
                const error = result.reason || {};
                const action = pollFailureAction(error.status, options.hasNonterminalTask());
                if (action === "missing") options.removeTask(id);
                else if (action === "auth") authError = error;
                else if (action === "retry") transientError = error;
            });

            if (successful && !transientError && !authError) poller.succeeded();
            options.saveTaskIds();
            options.render();
            if (authError) {
                poller.stop();
                options.onAuth(authError);
                return { action: "auth" };
            }
            if (transientError && options.hasNonterminalTask()) {
                options.onTransient(transientError);
                options.schedule(true, snapshot);
                return { action: "retry" };
            }
            if (options.hasNonterminalTask()) {
                options.schedule(false, snapshot);
                return { action: "poll" };
            }
            poller.stop();
            return { action: "stop" };
        }

        return { poll };
    }

    function pollFailureAction(status, nonterminal) {
        if (status === 401 || status === 403) return "auth";
        if (status === 404) return "missing";
        if (!nonterminal) return "stop";
        if (status == null || status >= 500) return "retry";
        return "stop";
    }

    function safeStorage(storage) {
        return {
            get(key) {
                try {
                    return { ok: true, value: storage ? storage.getItem(key) : null };
                } catch (_error) {
                    return { ok: false, value: null };
                }
            },
            set(key, value) {
                try {
                    if (!storage) return false;
                    storage.setItem(key, value);
                    return true;
                } catch (_error) {
                    return false;
                }
            },
            remove(key) {
                try {
                    if (!storage) return false;
                    storage.removeItem(key);
                    return true;
                } catch (_error) {
                    return false;
                }
            }
        };
    }

    function acceptSubmission(state, task, saveTaskIds, schedulePoll) {
        state.tasks[task.id] = task;
        if (state.taskIds.indexOf(task.id) === -1) state.taskIds.unshift(task.id);
        let stored = false;
        try {
            stored = saveTaskIds();
        } catch (_error) {
            stored = false;
        }
        schedulePoll();
        return stored;
    }

    return {
        lifecycleForStatus,
        createGenerationGate,
        createPollScheduler,
        createPollController,
        pollFailureAction,
        safeStorage,
        acceptSubmission
    };
}));
