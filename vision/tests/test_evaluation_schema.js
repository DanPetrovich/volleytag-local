"use strict";

const assert = require("node:assert/strict");
const evaluation = require("../../evaluation-schema.js");

function payload(overrides = {}) {
  return {
    schema: evaluation.SCHEMA,
    frozen_test: true,
    video: { duration_s: 10, width: 1920, height: 1080, roi: [40, 70] },
    inputs: { code_commit: "known-commit" },
    review: { complete: true },
    reviewed_ranges: [{ id: "r1", start_s: 0, end_s: 10 }],
    presence_intervals: [{ id: "p1", start_s: 1, end_s: 5, state: "visible", domain: "near_back" }],
    trajectory_intervals: [{
      id: "t1", presence_id: "p1", start_s: 1, end_s: 5, domain: "near_back",
      keyframes: [{ time_s: 1, box: [1, 2, 3, 4] }, { time_s: 5, box: [2, 3, 4, 5] }],
    }],
    validated_gaps: [], corrections: [], ...overrides,
  };
}

assert.equal(evaluation.validateEvaluation(payload()).valid, true, "complete valid evaluation");

const overlap = payload({ reviewed_ranges: [
  { id: "r1", start_s: 0, end_s: 6 }, { id: "r2", start_s: 5, end_s: 10 },
] });
assert.match(evaluation.validateEvaluation(overlap).errors.join("\n"), /overlap/);

const noGroundTruth = payload({ trajectory_intervals: [] });
assert.match(evaluation.validateEvaluation(noGroundTruth).errors.join("\n"), /not fully covered/);

const gapInsteadOfTrajectory = payload({
  trajectory_intervals: [],
  validated_gaps: [{ id: "g1", start_s: 1, end_s: 5, domain: "near_back", validated: true }],
});
assert.equal(evaluation.validateEvaluation(gapInsteadOfTrajectory).valid, true, "validated gap covers visible interval");

const incompleteCoverage = payload({ reviewed_ranges: [{ id: "r1", start_s: 0, end_s: 9 }] });
assert.match(evaluation.validateEvaluation(incompleteCoverage).errors.join("\n"), /100%/);

const uncertain = payload({
  presence_intervals: [{ id: "p1", start_s: 1, end_s: 5, state: "uncertain", domain: "far_front" }],
  trajectory_intervals: [],
});
assert.equal(evaluation.validateEvaluation(uncertain).valid, true, "uncertain is neither positive nor negative and needs no GT");

const trainingState = { anchors: [1], negatives: [2], lostEvents: [3], manualBoxes: [4] };
const exportedV3 = evaluation.trainingExport(trainingState, {
  video: "video", tracks: "tracks", roi: [40, 70], segments: [5],
});
assert.deepEqual(Object.keys(exportedV3), [
  "version", "video", "tracks", "roi", "domains", "segments", "anchors", "negatives", "lostEvents", "manualBoxes",
]);
assert.equal(exportedV3.version, 3);

console.log("evaluation-schema: 7 checks passed");
