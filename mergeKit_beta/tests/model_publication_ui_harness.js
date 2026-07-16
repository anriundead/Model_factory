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

console.log(`publication-ui harness: ${assertions} assertions passed`);
