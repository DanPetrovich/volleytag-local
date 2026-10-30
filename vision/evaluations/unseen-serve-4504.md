# Unseen serve rally check — failed identity test

- Original match timeline: 45:04–45:17
- Scenario: player #51 serves; this action type was not used to tune the two-rally acceptance run
- Profile: unchanged `player51-multiview.npz`
- ReID margin threshold: unchanged `0.10`
- OCR hits: 0
- Box coverage: 651 / 652 frames (99.8%), but this only measures whether *some* selected
  box is present and is not identity accuracy
- Selected track time: `8` 7.16 s, `87` 5.60 s, `82` 0.18 s, `63` 0.08 s
- Human audit: tracks `8` and `87` belong to another X5 Tech player near the net;
  player #51 starts in the back court serving and is selected only momentarily
- Result: **fail**

This held-out check exposed two failures. At the start of the rally the default 960 px YOLO
pass does not detect the distant player #51 at all, so the tracker has no correct trajectory
to follow. ReID then ranks a same-uniform teammate slightly above the true #51 fragments.
The previously reported 99.8% was therefore a false-positive coverage metric and must not be
treated as player-identification accuracy.

A 1536 px / 0.08 confidence diagnostic pass does detect #51 at the serve (temporary ID `25`),
but also produces duplicate and fragmented boxes. OCR still returns zero reliable `51` hits.
The next fix must address detection, duplicate suppression, and identity-aware stitching before
this rally can pass.
