# Results

**Competition:** [AI Agent Security — Multi-Step Tool Attacks](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks)
(hosted by OpenAI, Google, and IEEE; $50,000 prize pool)

**Final standing:** Bronze Medal — **290th of 4,186 teams (top 7%)**, solo entry.

**Winning submission:** `attack_v13_0_pure_cd_n800`
| Leaderboard | Score |
|---|---|
| Public | 9.925 |
| Private | 10.365 |

## Submission trajectory (selected)

| Version | Technique | N | Public | Private |
|---|---|---:|---:|---:|
| v9_1 | CONFUSED_DEPUTY, K=1 baseline | 200 | 4.105 | 4.105 |
| v13_0 | CONFUSED_DEPUTY, K=1 (winning) | 800 | 9.925 | 10.365 |
| v4_3 | EXFILTRATION, K=1, pure | 1200 | 0.000 | 40.675 |
| v6_8 | EXFILTRATION, K=1, pure (repeat) | 1200 | 0.000 | 41.075 |

Note the EXFILTRATION-based submissions score far higher on the *private* column but
**0.000 on public** — these were voided when the competition hardened its guardrails
mid-run. `CONFUSED_DEPUTY` (this submission's technique) survived on both, which is
the whole reason it — not the higher-raw-scoring EXFILTRATION runs — was the final
selected submission. See `experiments/lab_notebook.md` for the full experimental
history behind this decision.

Full leaderboard data available via the competition's public leaderboard export.
