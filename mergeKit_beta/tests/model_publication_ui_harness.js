"use strict";

const assert = require("assert");
const PublicationUI = require("../static/model_publication_ui.js");

let assertions = 0;
function equal(actual, expected, message) {
    assert.deepStrictEqual(actual, expected, message);
    assertions += 1;
}

const fixtures = {
    queued: [0, "active"],
    recipe: [0, "active"],
    materializing: [1, "active"],
    running: [1, "active"],
    validating: [2, "active"],
    registration_pending: [3, "active"],
    completed: [3, "completed"],
    failed: [null, "failed"],
    canceled: [null, "canceled"],
    unknown: [null, "unknown"]
};

Object.entries(fixtures).forEach(([status, expected]) => {
    const lifecycle = PublicationUI.lifecycleForStatus(status);
    equal(lifecycle.segment, expected[0], `${status} segment`);
    equal(lifecycle.kind, expected[1], `${status} kind`);
    equal(lifecycle.gateway, false, `${status} must not imply Gateway linkage`);
});

const generation = PublicationUI.createGenerationGate();
generation.changeToken("old-token");
const oldSnapshot = generation.snapshot();
generation.changeToken("new-token");
equal(generation.isCurrent(oldSnapshot), false, "old response must be stale");
equal(generation.isCurrent(generation.snapshot()), true, "current response must be accepted");

const timers = [];
const cleared = [];
const scheduler = PublicationUI.createPollScheduler({
    setTimer(callback, delay) {
        const timer = { callback, delay, id: timers.length + 1 };
        timers.push(timer);
        return timer;
    },
    clearTimer(timer) {
        cleared.push(timer.id);
    },
    normalDelay: 3000,
    retryBaseDelay: 1000,
    retryMaxDelay: 12000
});

scheduler.schedule(() => {}, false);
scheduler.schedule(() => {}, true);
equal(cleared, [1], "reschedule clears the existing timer");
equal(timers.map((timer) => timer.delay), [3000, 1000], "normal and first retry delays");
scheduler.schedule(() => {}, true);
scheduler.schedule(() => {}, true);
scheduler.schedule(() => {}, true);
scheduler.schedule(() => {}, true);
scheduler.schedule(() => {}, true);
equal(timers.slice(1).map((timer) => timer.delay), [1000, 2000, 4000, 8000, 12000, 12000], "retry backoff is bounded");
scheduler.succeeded();
scheduler.schedule(() => {}, true);
equal(timers[timers.length - 1].delay, 1000, "success resets retry backoff");

(async function runControllerIntegrationRegression() {
const controllerTimers = [];
const controllerCleared = [];
const controllerScheduler = PublicationUI.createPollScheduler({
    setTimer(callback, delay) {
        const timer = { callback, delay, id: controllerTimers.length + 1 };
        controllerTimers.push(timer);
        return timer;
    },
    clearTimer(timer) {
        controllerCleared.push(timer.id);
    },
    normalDelay: 3000,
    retryBaseDelay: 1000,
    retryMaxDelay: 4000
});
const controllerState = {
    tasks: {
        completed: { id: "completed", status: "running" },
        active: { id: "active", status: "running" }
    },
    ids: ["completed", "active"]
};
const controllerRefreshes = [];
const controllerCycles = [
    { completed: { id: "completed", status: "completed" }, active: { status: 503 } },
    { completed: { id: "completed", status: "completed" }, active: { network: true } },
    { completed: { id: "completed", status: "completed" }, active: { status: 503 } },
    { completed: { id: "completed", status: "completed" }, active: { id: "active", status: "running" } },
    { completed: { id: "completed", status: "completed" }, active: { status: 503 } }
];
let controllerCycleIndex = 0;
const controller = PublicationUI.createPollController({
    poller: controllerScheduler,
    isCurrent() { return true; },
    hasNonterminalTask() {
        return controllerState.ids.some((id) => PublicationUI.lifecycleForStatus(controllerState.tasks[id].status).kind === "active");
    },
    acceptTask(id, task) {
        controllerState.tasks[id] = task;
    },
    removeTask(id) {
        controllerState.ids = controllerState.ids.filter((taskId) => taskId !== id);
    },
    saveTaskIds() {},
    render() {},
    onAuth() {},
    onTransient() {},
    schedule(transient) {
        controllerScheduler.schedule(() => {}, transient);
    }
});

async function refreshControllerTask(id) {
    controllerRefreshes.push(id);
    const result = controllerCycles[controllerCycleIndex][id];
    if (result.network) throw new Error("network down");
    if (result.status && result.status >= 400) {
        const error = new Error(`HTTP ${result.status}`);
        error.status = result.status;
        throw error;
    }
    return result;
}

for (controllerCycleIndex = 0; controllerCycleIndex < controllerCycles.length; controllerCycleIndex += 1) {
    await controller.poll(controllerState.ids, refreshControllerTask);
    equal(controllerScheduler.hasTimer(), true, "mixed result cycle keeps one polling timer");
}
equal(controllerTimers.map((timer) => timer.delay), [1000, 2000, 4000, 3000, 1000], "controller preserves backoff across mixed cycles and resets after all-success cycle");
equal(controllerRefreshes, ["completed", "active", "completed", "active", "completed", "active", "completed", "active", "completed", "active"], "controller continues polling every active task");
equal(controllerCleared.length, 4, "controller replaces the shared timer once per cycle");

controllerCycleIndex = 0;
controllerCycles[0] = { completed: { id: "completed", status: "completed" }, active: { status: 401 } };
await controller.poll(controllerState.ids, refreshControllerTask);
equal(controllerScheduler.hasTimer(), false, "controller stops the shared timer on auth failure");

equal(PublicationUI.pollFailureAction(undefined, true), "retry", "network failure retries while active");
equal(PublicationUI.pollFailureAction(503, true), "retry", "5xx retries while active");
equal(PublicationUI.pollFailureAction(401, true), "auth", "401 requests token correction");
equal(PublicationUI.pollFailureAction(403, true), "auth", "403 requests token correction");
equal(PublicationUI.pollFailureAction(404, true), "missing", "404 stops missing task polling");
equal(PublicationUI.pollFailureAction(503, false), "stop", "terminal tasks do not retry");

const storage = PublicationUI.safeStorage({
    getItem() { throw new Error("get denied"); },
    setItem() { throw new Error("set denied"); },
    removeItem() { throw new Error("remove denied"); }
});
equal(storage.get("key"), { ok: false, value: null }, "get failure is contained");
equal(storage.set("key", "value"), false, "set failure is contained");
equal(storage.remove("key"), false, "remove failure is contained");

const submissionState = { taskIds: [], tasks: {} };
let scheduledPolls = 0;
const stored = PublicationUI.acceptSubmission(
    submissionState,
    { id: "task-storage-warning", status: "queued" },
    () => false,
    () => { scheduledPolls += 1; }
);
equal(stored, false, "submission reports only the storage warning");
equal(submissionState.taskIds, ["task-storage-warning"], "successful submission remains in memory");
equal(scheduledPolls, 1, "successful submission polls despite storage failure");

})().then(() => {
    console.log(`publication-ui harness: ${assertions} assertions passed`);
}).catch((error) => {
    console.error(error.stack || error);
    process.exitCode = 1;
});
