# Map or Ritual? Pre-registration of the confirmatory study (v1)

Locked on 2026-11-10 (IST), after the pilot and before any confirmatory episode was run.

## Frozen artifacts
- Gold labels: labels_v1.json, sha256 prefix a0ceb7ebce446eec
- Null worlds (W1 real normalized, W2 degree-preserving rewire, W3 permuted embeddings): manifest sha256 prefix 5b88c0e967618d8b, seed 20261108
- Both hashes were reproduced bit for bit by an independent re-execution (Kaggle commit).

## Population
- 82 primary tasks: at least one function-level gold node, no gold symbol name in the issue text, no gold hunk missing from the graph (including async), at most 8 source files changed.
- The 20 pilot tasks were drawn only from the 38 non-primary labeled tasks. No primary task was seen during development.

## Models
- gemma-4-e4b-it-qat-w4a16-ct and gemma-4-31b-it-qat-w4a16-ct, served by the official vLLM configuration (gemma4 tool and reasoning parsers, max_model_len 32768, temperature 0.7, per-request seed).

## Conditions (8), seeds {1, 2}
- NR: issue only, no tools
- W0: grep and read_file only
- W1, W2, W3: W0 tools plus the three graph tools, neutral prompt
- W1G, W2G, W3G: same worlds, graph-first instruction
- Budget 20 tool calls, then one forced submit. Requests time out after 400 s. Failed episodes are retried up to 2 more times.

## Outcomes
- Primary: hit@5, any of the top five predictions matches a gold node (exact, or dotted suffix with at least two components).
- Secondary: hit@1, file@5, tool calls, seconds, graph-call share, graph exposure (a gold node appears in a graph tool output).

## Hypotheses and tests
- H1 Adoption. Under the neutral prompt, the share of episodes with at least one graph call differs between models. Reported descriptively with task-clustered bootstrap 95% CIs.
- H2 Map value. In the graph-first arm, hit@5 in W1G exceeds W2G. Paired by task and seed. Mixed-effects logistic regression success ~ world + (1 | task), per model. Equivalence by TOST with margin 5 percentage points.
- H3 Mechanism. Any W1G minus W2G advantage is larger for graph-reachable tasks (hop 1 to 6 from an issue anchor) than for tasks with no anchor or no path. World by stratum interaction.
- H4 Embedding value. W1G versus W3G, same test as H2.
- H5 Same three contrasts in the neutral arm, for the model that adopts graph tools.
- Holm correction across H2, H3, H4 within each model.

## Handling rules
- Episodes still failing after retries count as misses. A sensitivity analysis excludes them.
- No tasks are added or removed after this point. Any deviation is reported as exploratory.
