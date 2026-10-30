# Serve rally 45:04–45:17 — appearance-gated result

- Scenario: player #51 starts in the distant back court and serves
- Detector: YOLO11n, image size 1536, confidence threshold 0.08, stride 2
- Confirmed identity anchors: `25:0` at the serve and `157:0` after the long occlusion
- Automatic chain: `25:0 → 44:1 → 60:0 → 60:1 → 27:1`, then `157:0`
- Minimum automatic continuation ReID margin: 0.0
- Rejected false continuation: yellow-jersey track `43:1`, margin `-0.00955`
- Short-gap interpolation: at most 0.8 s, only between selected geometrically compatible segments
- Selected-box coverage: 651 / 652 frames (99.8%)
- Identity audit: crops sampled every 0.25 s plus dedicated samples inside both interpolated gaps
- Observed wrong-player duration in the audit: 0 s
- Result: **pass for this rally in semi-automatic mode**

The critical behavioral change is abstention: an automatic continuation with a negative ReID
margin is no longer accepted just because it is nearby. This prevents the former 5.28-second
switch to the woman warming up in a yellow jersey. Two visually confirmed segments provide the
identity on both sides of the difficult portion; the intermediate fragments are selected by
geometry only after passing the appearance gate.

This does not yet prove automatic whole-match identification. The two seed segments were
confirmed manually, and the same acceptance audit must be repeated on additional held-out
rallies before relaxing that requirement.
