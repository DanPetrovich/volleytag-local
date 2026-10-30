(function initVolleyTagEvaluation(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.VolleyTagEvaluation = api;
}(typeof globalThis !== "undefined" ? globalThis : this, () => {
  "use strict";

  const SCHEMA = "volleytag.player51-evaluation.v4";
  const PRESENCE_STATES = ["visible", "occluded", "out_of_frame", "uncertain"];
  const DOMAINS = ["near_back", "far_front"];

  function number(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function intervalBounds(interval) {
    return [number(interval?.start_s), number(interval?.end_s)];
  }

  function intervalErrors(intervals, label, { allowOverlap = false } = {}) {
    const errors = [];
    const ordered = intervals.map((interval, index) => ({ interval, index, bounds: intervalBounds(interval) }))
      .sort((left, right) => (left.bounds[0] ?? Infinity) - (right.bounds[0] ?? Infinity));
    ordered.forEach(({ index, bounds }) => {
      const [start, end] = bounds;
      if (start === null || end === null || start < 0 || end <= start) {
        errors.push(`${label}[${index}] must have finite 0 <= start_s < end_s`);
      }
    });
    if (!allowOverlap) {
      for (let index = 1; index < ordered.length; index += 1) {
        const previous = ordered[index - 1], current = ordered[index];
        if (previous.bounds[1] !== null && current.bounds[0] !== null
            && current.bounds[0] < previous.bounds[1] - 1e-9) {
          errors.push(`${label} intervals overlap at indexes ${previous.index} and ${current.index}`);
        }
      }
    }
    return errors;
  }

  function mergeRanges(ranges) {
    const valid = ranges.map(intervalBounds).filter(([start, end]) => start !== null && end !== null && end > start)
      .sort((left, right) => left[0] - right[0]);
    const merged = [];
    valid.forEach(([start, end]) => {
      const previous = merged.at(-1);
      if (!previous || start > previous[1] + 1e-9) merged.push([start, end]);
      else previous[1] = Math.max(previous[1], end);
    });
    return merged;
  }

  function coveredSeconds(ranges, clipStart, clipEnd) {
    if (!Number.isFinite(clipStart) || !Number.isFinite(clipEnd) || clipEnd <= clipStart) return 0;
    return mergeRanges(ranges).reduce((total, [start, end]) => (
      total + Math.max(0, Math.min(end, clipEnd) - Math.max(start, clipStart))
    ), 0);
  }

  function reviewedCoverage(reviewedRanges, duration) {
    const total = number(duration);
    if (total === null || total <= 0) return { seconds: 0, ratio: 0 };
    const seconds = coveredSeconds(reviewedRanges, 0, total);
    return { seconds, ratio: Math.min(1, seconds / total) };
  }

  function storageKey(videoName, tracksName) {
    return `volleytag-evaluation-v4:${videoName || ""}:${tracksName || ""}`;
  }

  function shouldRestoreQueryEvaluation(mode, videoUrl, tracksUrl) {
    return mode === "evaluation" && Boolean(videoUrl) && Boolean(tracksUrl);
  }

  function fullyCovered(start, end, ranges) {
    const clipped = mergeRanges(ranges.map((range) => ({
      start_s: Math.max(start, Number(range.start_s)),
      end_s: Math.min(end, Number(range.end_s)),
    })).filter((range) => Number.isFinite(range.start_s) && Number.isFinite(range.end_s)
      && range.end_s > range.start_s));
    if (!clipped.length || clipped[0][0] > start + 1e-6) return false;
    let cursor = clipped[0][1];
    for (let index = 1; index < clipped.length && cursor < end - 1e-6; index += 1) {
      if (clipped[index][0] > cursor + 1e-6) return false;
      cursor = Math.max(cursor, clipped[index][1]);
    }
    return cursor >= end - 1e-6;
  }

  function validateEvaluation(payload) {
    const errors = [], warnings = [];
    if (!payload || typeof payload !== "object") return { valid: false, errors: ["evaluation payload must be an object"], warnings };
    if (payload.schema !== SCHEMA) errors.push(`schema must equal ${SCHEMA}`);
    if (payload.frozen_test !== true) errors.push("frozen_test must be true");
    const reviewed = Array.isArray(payload.reviewed_ranges) ? payload.reviewed_ranges : [];
    const presence = Array.isArray(payload.presence_intervals) ? payload.presence_intervals : [];
    const trajectories = Array.isArray(payload.trajectory_intervals) ? payload.trajectory_intervals : [];
    const gaps = Array.isArray(payload.validated_gaps) ? payload.validated_gaps : [];
    const corrections = Array.isArray(payload.corrections) ? payload.corrections : [];
    errors.push(...intervalErrors(reviewed, "reviewed_ranges"));
    errors.push(...intervalErrors(presence, "presence_intervals"));
    errors.push(...intervalErrors(trajectories, "trajectory_intervals", { allowOverlap: true }));
    errors.push(...intervalErrors(gaps, "validated_gaps", { allowOverlap: true }));
    presence.forEach((interval, index) => {
      if (!PRESENCE_STATES.includes(interval.state)) errors.push(`presence_intervals[${index}].state is invalid`);
      if (!DOMAINS.includes(interval.domain)) errors.push(`presence_intervals[${index}].domain is invalid`);
      if (interval.state !== "visible") return;
      const [start, end] = intervalBounds(interval);
      if (start === null || end === null || end <= start) return;
      const usableTrajectories = trajectories.filter((trajectory) =>
        Array.isArray(trajectory.keyframes) && trajectory.keyframes.length > 1
        && trajectory.domain === interval.domain).map((trajectory) => {
        const times = trajectory.keyframes.map((keyframe) => number(keyframe.time_s)).filter((time) => time !== null);
        return { start_s: Math.min(...times), end_s: Math.max(...times) };
      });
      const validatedGaps = gaps.filter((gap) => gap.validated === true && gap.domain === interval.domain);
      if (!fullyCovered(start, end, [...usableTrajectories, ...validatedGaps])) {
        errors.push(`visible presence_intervals[${index}] is not fully covered by human GT trajectory or validated gap`);
      }
    });
    trajectories.forEach((trajectory, index) => {
      if (!DOMAINS.includes(trajectory.domain)) errors.push(`trajectory_intervals[${index}].domain is invalid`);
      if (!Array.isArray(trajectory.keyframes)) errors.push(`trajectory_intervals[${index}].keyframes must be an array`);
      else trajectory.keyframes.forEach((keyframe, keyframeIndex) => {
        const box = keyframe.box;
        if (number(keyframe.time_s) === null || !Array.isArray(box) || box.length !== 4
            || box.some((value) => number(value) === null)) {
          errors.push(`trajectory_intervals[${index}].keyframes[${keyframeIndex}] is invalid`);
        }
      });
    });
    corrections.forEach((correction, index) => {
      const start = number(correction.error_start_s), end = number(correction.corrected_at_s);
      if (start === null || end === null || start < 0 || end < start) {
        errors.push(`corrections[${index}] must have 0 <= error_start_s <= corrected_at_s`);
      }
      if (!DOMAINS.includes(correction.domain)) errors.push(`corrections[${index}].domain is invalid`);
      if (typeof correction.reason !== "string" || !correction.reason.trim()) errors.push(`corrections[${index}].reason is required`);
      ["before", "after"].forEach((side) => {
        if (!correction[side] || !Array.isArray(correction[side].box) || correction[side].box.length !== 4
            || !correction[side].identity) errors.push(`corrections[${index}].${side} identity and box are required`);
      });
    });
    const duration = number(payload.video?.duration_s);
    if (duration === null || duration <= 0) errors.push("video.duration_s must be a positive number");
    const coverage = reviewedCoverage(reviewed, duration);
    if (payload.review?.complete === true && coverage.ratio < 1 - 1e-6) {
      errors.push("review.complete requires 100% reviewed coverage");
    }
    if (payload.review?.complete !== true) warnings.push("review is not marked complete");
    if (!payload.inputs || !Object.keys(payload.inputs).length) warnings.push("input hashes/commit metadata are not available");
    return { valid: errors.length === 0, errors, warnings, coverage };
  }

  function trainingExport(state, metadata) {
    return {
      version: 3,
      video: metadata.video,
      tracks: metadata.tracks,
      roi: metadata.roi,
      domains: { near_back: "ближняя сторона, спиной", far_front: "дальняя сторона, лицом" },
      segments: metadata.segments,
      anchors: state.anchors,
      negatives: state.negatives,
      lostEvents: state.lostEvents,
      manualBoxes: state.manualBoxes,
    };
  }

  function emptyEvaluation() {
    return {
      reviewed_ranges: [], presence_intervals: [], trajectory_intervals: [],
      validated_gaps: [], corrections: [],
      review: { complete: false, annotator: "", notes: [] },
      inputs: {}, activePresenceId: null, activeReviewedId: null, activeGapId: null,
    };
  }

  return {
    SCHEMA, PRESENCE_STATES, DOMAINS, emptyEvaluation, intervalErrors,
    reviewedCoverage, shouldRestoreQueryEvaluation, storageKey, trainingExport, validateEvaluation,
  };
}));
