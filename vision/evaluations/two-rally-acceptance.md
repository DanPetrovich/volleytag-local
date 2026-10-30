# Two-rally acceptance check

Checked against the original match timeline. The target is player #51.

| Clip | Automatic selection | Coverage | Audited wrong-player time | Result |
|---|---|---:|---:|---|
| Set 1, 06:45–07:20 | OCR anchor + continuous chunks `7 → 249 → 261 → 259` | 98.9% | 0.0 s | pass |
| Set 2, 25:40–25:57 | strict ReID top-1 chunks `5 → 39 → 88` | 95.1% | 0.0 s | pass |

The selected chunks were visually audited from dense contact sheets and boundary frames. Blank
frames count against coverage. A ReID similarity margin of `0.10` is used for the second clip;
lower-scored intervals are deliberately left blank instead of switching to another player.

This is an acceptance result for these two clips, not a claim of the same accuracy on an unseen
match. The threshold was selected during work on this footage and must be revalidated on another
match before being treated as a general default.

Regression check: the handshake clip that previously attached tracker ID 37 to both #57 and #51
is now split at observation gaps. Only continuous chunk `37:5` (09.44–20.00) receives the OCR
identity; the earlier #57 chunks are excluded.
