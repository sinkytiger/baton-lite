# Results (test split, 30 tasks)

## claude-sonnet-5-5 → qwen3:4b

| Condition | Transfer | Correct | Accuracy % (95% CI) | API $ per task | Cost vs capable-only | Handoff turn / length | vs capable-only, pts (95% CI) | McNemar p |
|---|---|---|---|---|---|---|---|---|
| capable | - | 30/30 | 100.0 (88.6–100.0) | 0.0220 | +0% | - | - | - |
| local | - | 0/30 | 0.0 (0.0–11.4) | 0.0000 | -100% | - | -100.0 (-100.0 to -100.0) | 0.000 |
| fixed25 | incontext | 8/30 | 26.7 (14.2–44.4) | 0.0032 | -85% | 28% (30/30 handed off) | -73.3 (-86.7 to -56.7) | 0.000 |
| fixed50 | incontext | 12/30 | 40.0 (24.6–57.7) | 0.0092 | -58% | 49% (30/30 handed off) | -60.0 (-76.7 to -43.3) | 0.000 |
| fixed75 | incontext | 18/30 | 60.0 (42.3–75.4) | 0.0161 | -27% | 72% (30/30 handed off) | -40.0 (-56.7 to -23.3) | 0.000 |
| rule | incontext | 26/30 | 86.7 (70.3–94.7) | 0.0185 | -16% | 78% (30/30 handed off) | -13.3 (-26.7 to -3.3) | 0.125 |
| prompt | incontext | 16/30 | 53.3 (36.1–69.8) | 0.0117 | -47% | 58% (30/30 handed off) | -46.7 (-63.3 to -30.0) | 0.000 |
| fixed50 | logfile | 2/30 | 6.7 (1.8–21.3) | 0.0092 | -58% | 49% (30/30 handed off) | -93.3 (-100.0 to -83.3) | 0.000 |
| rule | logfile | 25/30 | 83.3 (66.4–92.7) | 0.0185 | -16% | 78% (30/30 handed off) | -16.7 (-30.0 to -3.3) | 0.062 |
| prompt | logfile | 9/30 | 30.0 (16.7–47.9) | 0.0117 | -47% | 58% (30/30 handed off) | -70.0 (-86.7 to -53.3) | 0.000 |
| oracle | incontext | 30/30 | 100.0 (88.6–100.0) | 0.0110 | -50% | 55% (29/30 handed off) | +0.0 (+0.0 to +0.0) | 1.000 |
| reverse (qwen3:4b → claude-sonnet-5-5) at 50% | incontext | 29/30 | 96.7 (83.3–99.4) | 0.0335 | +52% | - | -3.3 (-10.0 to +0.0) | 1.000 |

Handoff curve (in-context): share of continuations that end correct, by when the handoff happens

| Handoff after | continuations | correct % |
|---|---|---|
| 0–25% of the capable run | 29 | 13.8 |
| 25–50% of the capable run | 40 | 27.5 |
| 50–75% of the capable run | 51 | 47.1 |
| 75–100% of the capable run | 32 | 90.6 |

Prompt detector (152 judgements on 30 tasks): READY 72, NOT_READY 80, unparsable 0; fires at turn 1 on 0 tasks, never on 0.

Replay mismatches (prefix did not reproduce the recorded workspace): 0 of 265.

Wrong in-context continuations by cause: wrong 47, no_result 27, no_restatement 7, all_millions 2, unit_slip 1

## claude-haiku-5-5 → qwen3:4b

| Condition | Transfer | Correct | Accuracy % (95% CI) | API $ per task | Cost vs capable-only | Handoff turn / length | vs capable-only, pts (95% CI) | McNemar p |
|---|---|---|---|---|---|---|---|---|
| capable | - | 29/30 | 96.7 (83.3–99.4) | 0.0018 | +0% | - | - | - |
| local | - | 0/30 | 0.0 (0.0–11.4) | 0.0000 | -100% | - | -96.7 (-100.0 to -90.0) | 0.000 |
| fixed25 | incontext | 7/30 | 23.3 (11.8–40.9) | 0.0002 | -88% | 26% (30/30 handed off) | -73.3 (-86.7 to -56.7) | 0.000 |
| fixed50 | incontext | 7/30 | 23.3 (11.8–40.9) | 0.0006 | -69% | 46% (30/30 handed off) | -73.3 (-86.7 to -56.7) | 0.000 |
| fixed75 | incontext | 19/30 | 63.3 (45.5–78.1) | 0.0013 | -29% | 74% (30/30 handed off) | -33.3 (-50.0 to -16.7) | 0.002 |
| rule | incontext | 27/30 | 90.0 (74.4–96.5) | 0.0016 | -12% | 82% (30/30 handed off) | -6.7 (-16.7 to +0.0) | 0.500 |
| prompt | incontext | 11/30 | 36.7 (21.9–54.5) | 0.0009 | -51% | 60% (29/30 handed off) | -60.0 (-76.7 to -43.3) | 0.000 |
| fixed50 | logfile | 0/30 | 0.0 (0.0–11.4) | 0.0006 | -69% | 46% (30/30 handed off) | -96.7 (-100.0 to -90.0) | 0.000 |
| rule | logfile | 25/30 | 83.3 (66.4–92.7) | 0.0016 | -12% | 82% (30/30 handed off) | -13.3 (-26.7 to -3.3) | 0.125 |
| prompt | logfile | 5/30 | 16.7 (7.3–33.6) | 0.0009 | -51% | 60% (29/30 handed off) | -80.0 (-93.3 to -66.7) | 0.000 |
| oracle | incontext | 29/30 | 96.7 (83.3–99.4) | 0.0011 | -38% | 59% (27/30 handed off) | +0.0 (+0.0 to +0.0) | 1.000 |

Handoff curve (in-context): share of continuations that end correct, by when the handoff happens

| Handoff after | continuations | correct % |
|---|---|---|
| 0–25% of the capable run | 30 | 20.0 |
| 25–50% of the capable run | 34 | 23.5 |
| 50–75% of the capable run | 46 | 28.3 |
| 75–100% of the capable run | 30 | 90.0 |

Prompt detector (140 judgements on 30 tasks): READY 67, NOT_READY 73, unparsable 0; fires at turn 1 on 0 tasks, never on 1.

Replay mismatches (prefix did not reproduce the recorded workspace): 0 of 260.

Wrong in-context continuations by cause: wrong 57, no_result 27, all_millions 1, all_thousands 1
