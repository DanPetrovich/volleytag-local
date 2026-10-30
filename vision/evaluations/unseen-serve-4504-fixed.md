# Serve rally 45:04–45:17 — failed seeded high-resolution attempt

- Scenario: player #51 starts in the distant back court and serves
- Detector: YOLO11n, image size 1536, confidence threshold 0.08, stride 2
- Confirmed identity anchor: continuous chunk `25:0` at 0.00–1.72 s
- Stitched chain: `25:0 → 44:1 → 60:0 → 43:1 → 157:0`
- Selected-box coverage: 633 / 652 frames (97.1%)
- Largest gap between selected segments: 0.24 s
- Identity audit correction: chunk `43:1` (2.68–7.96 s) is a woman in a yellow jersey
  warming up behind the court, not player #51
- Wrong-player duration: approximately 5.28 s
- Result: **fail**

The higher-resolution detector does fix the original first-stage failure: it detects #51 while
he is still small and serving from the back court. However, geometry-only continuation makes an
identity switch after track `60:0`. Track `43:1` begins only 0.08 s later and 35 px away, so the
fusion stage accepts it even though its ReID margin is negative (`-0.00955`). The current logic
always chooses the best geometrically valid candidate and has no abstain state.

The earlier claim that 27 / 27 sampled crops showed #51 was a human review error: the yellow
jersey is visible in the saved audit sheet. Coverage is therefore not identity accuracy. A safe
fix must reject appearance-incompatible continuations and leave the frame empty when identity is
uncertain; only then should it attempt later reacquisition.
