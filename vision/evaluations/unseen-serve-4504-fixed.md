# Serve rally 45:04–45:17 — seeded high-resolution fix

- Scenario: player #51 starts in the distant back court and serves
- Detector: YOLO11n, image size 1536, confidence threshold 0.08, stride 2
- Confirmed identity anchor: continuous chunk `25:0` at 0.00–1.72 s
- Stitched chain: `25:0 → 44:1 → 60:0 → 43:1 → 157:0`
- Selected-box coverage: 633 / 652 frames (97.1%)
- Largest gap between selected segments: 0.24 s
- Identity audit: 27 crops sampled every 0.5 s; 27 / 27 show player #51
- Observed wrong-player duration: 0 s in the sampled audit
- Result: **pass for this rally**

The higher-resolution detector fixes the original first-stage failure: it detects #51 while he
is still small and serving from the back court. A confirmed continuous chunk is used as the
identity anchor. The fusion stage then follows a single geometrically continuous chain, so
overlapping detector candidates do not produce multiple final boxes.

This is a semi-automatic result: chunk `25:0` was visually confirmed. It demonstrates that the
tracking and stitching stages work once the distant player is detected and anchored; automatic
serve-player discovery remains a separate task.
