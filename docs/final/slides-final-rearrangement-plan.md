# Final Slides Rearrangement Plan

Purpose: approval doc for rearranging `docs/final/slides-final.tex`.

Audience: systems professor. The deck should explain RL as an engineering response to an online systems allocation problem, not as an RL tutorial. Use RL vocabulary only after the systems failure mode is clear.

Current source of truth:
- `docs/final/slides-final.tex`: current deck and existing TikZ/figure assets.
- `docs/training/v4/reward-shaping.tex`: latest reward ablation, optimal-gap framing, and held-out results.
- `docs/data_augmentation/data_aug.tex`: trace characterization and augmentation knob selection.
- `docs/plans/v4/throughput_log.md`: measured throughput before/after system redesign.
- `docs/plans/v4/speedup-plan.md`: system redesign rationale.
- `docs/plans/v3/async-plan.md`: async eval worker design.
- `docs/training/v3/metrics-guide.tex`: SAC explanation, collapse diagnostics, and reward failure modes.

Important approval decision: the current final deck says RL beats greedy on both held-out traces with ratios `0.98 / 1.06` versus greedy `3.77 / 12.97`. The newer reward-shaping doc reports a more nuanced result: on LVL01, best RL `R2` beats greedy (`1.002` versus `1.033`), but on AMS20 greedy beats all RL variants (`0.721` versus RL around `0.934`). I recommend using the newer nuanced result for credibility unless the old comparison was produced under a different finalized evaluation setup that we should preserve. -- agreed

## Proposed Narrative

1. Why consider RL at all?
2. What is the best result against greedy?
3. How did we get there?
4. What remains unsolved?

The story should be: greedy is strong but local; exact optimality is useful as a lower bound but not deployable as an online policy with future arrivals; RL was worth trying because it can learn trace-level placement patterns and generalize across pods/traces. The strongest current conclusion is not "RL always wins"; it is "RL can beat greedy on churn-heavy traces, but the remaining problem is distribution-aware generalization and reward alignment."

## Slide 0: Title

- [ ] Approved
- [ ] Needs edits

Title: `RL for CXL Memory Pooling: When Learning Helps, When Greedy Wins`

Bullets: none.

Diagram: none.

Speaker intent: signal that the talk is an honest systems result, not a pure ML victory lap.

Source: current title slide in `slides-final.tex`.

## Slide 1: Why Did We Even Consider RL?

- [ ] Approved
- [ ] Needs edits

Working title: `Why RL was worth trying: greedy is local, the objective is global.`

Bullets to design:
- The system receives VMs online and must place each memory request on one reachable MPD.
- Greedy only sees current load and picks the least-loaded reachable MPD.
- The actual objective is episode-level peak MPD load over the whole trace.
- A placement that looks good now can make a shared MPD unusable for another host later.
- RL is attractive because it can condition on state features that summarize departures, future committed load, and pod pressure.

Diagram to design:
- Left: bipartite host-to-MPD topology. Highlight one arriving VM whose host can only reach a local MPD neighborhood.
- Right: global peak objective across all MPDs. Show that the local action changes a shared resource used by future hosts.
- Use a small callout: `local action -> shared resource -> global peak`.

Professor-facing explanation:
- Avoid starting with "MDP" or "policy gradient."
- Start with "online placement under shared-resource interference."
- Then say RL is a way to learn a placement rule from trace replay when the best local rule is not obviously greedy.

Sources:
- `docs/training/v4/reward-shaping.tex`: reachability and reward scope problem.
- `docs/final/slides-final.tex`: current "Greedy is myopic" slide.

## Slide 2: Why Not Just Greedy Or Optimal?

- [ ] Approved
- [ ] Needs edits

Working title: `Greedy is deployable; optimal is a benchmark; RL tries to close the gap.`

Bullets to design:
- Greedy is cheap and online, so it is the real baseline.
- The exact solver gives a per-snapshot lower bound via max-flow, but it is not the same as solving the full future-arrival problem.
- The gap between greedy and the lower bound tells us whether there is room for a smarter policy.
- RL was considered only because the trace has repeatable structure: lifetimes, churn, pod heterogeneity, and shared-MPD contention.

Diagram to design:
- Three-column ladder:
  - `Greedy`: online, local, cheap, myopic.
  - `Optimal lower bound`: global snapshot, expensive, not future-aware.
  - `RL policy`: online at inference, trained offline on traces, can use departure-aware features.
- Add a small "gap to close" brace between greedy and optimal/RL.

Professor-facing explanation:
- Be precise: "optimal" here is a lower-bound / diagnostic tool, not a deployable oracle for the whole 14-day trace.
- The slide should justify the experiment without overselling RL.

Sources:
- `docs/training/v4/reward-shaping.tex`: optimal gap, Dinic max-flow, lower-bound framing.
- `docs/training/v4/reward-shaping.tex`: "A 1--2% improvement over greedy is considered strong."

## Slide 3: Best RL Result Compared To Greedy

- [ ] Approved
- [ ] Needs edits

Working title: `Best result: RL helps on churn-heavy traces, but generalization is not solved.`

Bullets to design:
- Latest held-out result: on LVL01, greedy ratio is `1.033`; best RL (`R2`) reaches `1.002`.
- That is a `+3.1 percentage point` improvement in savings and nearly break-even.
- On AMS20, greedy ratio is `0.721`; RL variants cluster near `0.934`, so greedy wins by about `21 percentage points`.
- Interpretation: RL learned behavior that transfers to churn-heavy traces, but not to the long-lived AMS20 regime.
- This makes the core remaining issue a systems generalization problem, not just a bigger-network problem.

make the graphs using matplotlib

Diagram to design:
- Grouped bar chart with two traces: LVL01 and AMS20.
- Bars: Greedy, best RL, maybe "current reward" if space permits.
- Use lower-is-better pool ratio on the left axis, and annotate `break-even = 1.0`.
- Add trace labels: `LVL01: churn-heavy` and `AMS20: long-lived outlier`.

Alternative if older results are the approved final numbers:
- Use the existing `comparison_bar.pdf` story: RL `0.98 / 1.06` versus greedy `3.77 / 12.97`. -- no
- Then remove the AMS20 generalization-gap claim from this slide.

Sources:
- Preferred: `docs/training/v4/reward-shaping.tex` held-out tables and cross-trace summary.
- Alternative: current `docs/final/slides-final.tex` and `docs/overview/v4/slides/slides-v4.tex`.

## Slide 4: How We Got There: Choose The RL Algorithm For Stability

- [ ] Approved
- [ ] Needs edits

Working title: `SAC was chosen because collapse was the first failure mode.`

Bullets to design:
- Early policies collapsed: the agent repeatedly selected one or two MPDs even when load changed.
- SAC adds an entropy term, which keeps the policy from becoming deterministic too early.
- Replay-buffer off-policy learning fits trace replay: the same allocation experience can be reused for many gradient updates.
- SB3 SAC worked cleanly with 32 parallel environments, which mattered more than theoretical elegance.
- Docs support comparisons against LCPO and A2C; PPO-specific evidence is not currently in the docs and needs confirmation before making a PPO claim.

Diagram to design:
- Simple control-loop diagram:
  - `32 env rollouts -> replay buffer -> critic update -> entropy-regularized actor -> policy`.
- Next to it, a tiny contrast:
  - `without entropy: collapse to one MPD`
  - `with entropy: keep exploring plausible MPDs`

Professor-facing explanation:
- Explain entropy as an engineering guardrail against premature load-concentration.
- Avoid a deep actor-critic derivation unless asked.
- If we mention PPO, phrase carefully: "we chose SAC over on-policy alternatives because off-policy replay and entropy regularization matched our trace-replay setup." Do not claim measured PPO results unless available.

Sources:
- `docs/training/v3/metrics-guide.tex`: SAC internals and entropy explanation.
- `docs/final/slides-final.tex`: LCPO/A2C instability and SAC chosen.

Approval question:
- Should this slide say `SAC vs PPO`, or should it say `SAC vs on-policy alternatives` to avoid unsupported PPO-specific claims?

## Slide 5: Data Augmentation: Knobs Were Chosen From Trace Constraints

- [ ] Approved
- [ ] Needs edits

Working title: `Augmentation was domain randomization, not random noise.`

Bullets to design:
- Randomize at the episode level because VM memory sizes are SKU-like; per-VM scaling would create fake VM types.
- Use three enabled knobs: trace ID, pod seed, and memory scale.
- Keep memory scale in `[0.7, 1.2]` because lower values add little diversity and higher values cause excessive HOTFIX dropout.
- Use triangular scale centered at `1.0` to stay near realistic load while still covering light/heavy regimes.
- Leave arrival jitter, lifetime perturbation, and link failures off unless we explicitly validate them.

Diagram to design:
- "Knob board" with green enabled knobs and gray disabled knobs:
  - Enabled: `trace`, `pod seed`, `episode memory scale`.
  - Disabled: `arrival jitter`, `lifetime perturbation`, `link failures`.
- Add a small survival-rate mini-plot or callout:
  - `scale 0.7 -> 83.6% survival`
  - `scale 1.0 -> 81.0%`
  - `scale 1.2 -> 76.7%`
  - `scale 1.5 -> 64.9%`

Professor-facing explanation:
- This is not ML data augmentation in image-space terms.
- It is simulator domain randomization constrained by production trace validity.

Sources:
- `docs/data_augmentation/data_aug.tex`: SKU structure, HOTFIX survival, safe scale range, augmentation knobs.
- Current `docs/final/slides-final.tex`: existing augmentation diagram.

## Slide 6: Did Data Augmentation Help?

- [ ] Approved
- [ ] Needs edits

Working title: `Augmentation improved robustness, but did not solve lifetime-distribution shift.`

Bullets to design:
- Single-trace training overfit and failed on unseen traces.
- Multi-trace + pod-seed + scale augmentation made training stable and enabled held-out evaluation. -- are there any results i can add for this? compare run 7 and other runs.
- The newest results suggest the augmentation distribution is still biased: 9 of 10 traces are churn-dominated, while AMS20 is long-lived.
- Therefore augmentation helped, but its next version must balance lifetime regimes, not just sample traces uniformly.

Diagram to design:
- Two-panel diagram:
  - Left: before augmentation, single trace -> brittle policy.
  - Right: after augmentation, trace pool + pod seeds + scale -> robust on churn-heavy held-out trace.
- Optional small table of lifetime distributions:
  - `AMS20`: median `124.6h`, `49.3% > 7d`.
  - `LVL01`: median `0.75h`, `56.7% < 1h`.

Professor-facing explanation:
- This is the bridge from "RL result" to "systems generalization."
- Emphasize that the failure is interpretable from trace statistics.

Sources:
- `docs/training/v4/reward-shaping.tex`: cross-trace summary and lifetime table.
- `docs/data_augmentation/data_aug.tex`: trace inventory and augmentation design.
- Current `docs/final/slides-final.tex`: "single-trace policy fails; augmentation fixes it" claim, subject to updated-results wording.

## Slide 7: Reward Design: Proxy, Terminal, Optimal

- [ ] Approved
- [ ] Needs edits

Working title: `Reward design is about aligning local decisions with the global peak objective.`

Bullets to design:
- Proxy rewards are dense and easy to learn but can optimize the wrong thing.
- Terminal reward is exactly aligned with pooling savings but sparse over thousands of placement decisions.
- Optimal-gap shaping uses a max-flow lower bound to give a principled dense signal, but it has compute cost.
- The ablation ladder asks which amount of global information is necessary: single MPD, reachable neighborhood, whole pod, full episode, or optimal-gap shaping.
- Early R1/R2/R3 results plateau near `~5%` in training; held-out reward variants are often indistinguishable, so reward alone may not solve the generalization gap.

Diagram to design:
- Horizontal ladder:
  - `R1 single worst MPD`
  - `R2 reachable MPDs`
  - `R3 global proxy`
  - `R4 terminal true objective`
  - `R5 PBRS + sub-episodes`
  - `optimal gap`
- Under each rung, show two tags: `dense/sparse` and `aligned/proxy`.
- Use color to mark the tradeoff: red for proxy risk, green for objective alignment, orange for credit-assignment cost.

Professor-facing explanation:
- Do not present reward design as tuning.
- Present it as a measurement problem: "what scalar signal faithfully measures the systems objective at each decision?"

Sources:
- `docs/training/v4/reward-shaping.tex`: ablation hypothesis, R4/R5 definitions, optimal gap.
- `docs/new-results/ablation-r1r2r3-analysis.md`: R1/R2/R3 plateau and diagnostics.
- Current `docs/final/slides-final.tex`: existing reward-regime slide.

## Slide 8: New System Design: Throughput Made The Science Possible

- [ ] Approved
- [ ] Needs edits

Working title: `The systems redesign turned reward ablation from days into hours.`

Bullets to design:
- The bottleneck was CPU environment stepping and synchronous evaluation, not the GPU.
- Async evaluation removed about `44 min` of blocked training time per 2M-step run.
- Flat MPD arrays replaced list-of-lists hot loops.
- Multi-trace precompute cache reduced reset overhead.
- Numba JIT compiled the departure and sticky-load kernels.
- Env-only throughput improved from `1,009 fps` pre-speedup to `2,757 fps` post-speedup; compared with the old full-training baseline of `~53 fps`, the env is no longer the limiting factor.

Diagram to design:
- Left: pipeline before: `train -> blocked eval -> train -> blocked eval`.
- Right: pipeline after: `train continuously` with eval worker running below it.
- Add a throughput bar chart:
  - `run7 full training baseline: 53 fps`
  - `pre-speedup env-only: 1,009 fps`
  - `post-speedup env-only: 2,757 fps`
- Add labels for the three engineering changes: `flat arrays`, `cache`, `Numba`.

Professor-facing explanation:
- This slide should appeal directly to a systems professor: the result was only possible because the simulator became fast enough to run controlled ablations.
- Be careful comparing full-training fps and env-only fps; label them explicitly.

Sources:
- `docs/plans/v4/throughput_log.md`: measured fps.
- `docs/plans/v4/speedup-plan.md`: flat arrays, cache, Numba rationale.
- `docs/plans/v3/async-plan.md`: 44 min blocked eval.
- Current `docs/final/slides-final.tex`: existing engineering slide.

## Slide 9: What We Learned

- [ ] Approved
- [ ] Needs edits

Working title: `What the experiments actually say.`

Bullets to design:
- RL is not automatically better than greedy; it helps when the trace structure rewards departure-aware placement.
- Greedy remains strong on long-lived workloads.
- The largest remaining gap is distribution-aware generalization across VM lifetime regimes.
- Reward shaping made the problem measurable, but did not by itself solve cross-trace transfer.
- The training system is now fast enough to answer the next questions scientifically.

Diagram to design:
- Three takeaway cards:
  - `Where RL helps`: high churn, departure timing matters.
  - `Where greedy wins`: long-lived stable load.
  - `Next bottleneck`: balanced training distribution + aligned reward.

Professor-facing explanation:
- This is the honest synthesis slide.
- It should replace any overbroad "RL solves CXL pooling" summary.

Sources:
- `docs/training/v4/reward-shaping.tex`: cross-trace summary.
- `docs/final/slides-final.tex`: existing summary slide, revised for newer interpretation.

## Slide 10: Future Directions

- [ ] Approved
- [ ] Needs edits

Working title: `Future work: make the learned policy robust, not just better on one regime.`

Bullets to design:
- Balance training batches by lifetime regime so AMS20-like long-lived traces are not underrepresented.
- Add explicit optimal-gap diagnostics to separate "bad policy" from "hard instance."
- Expand data augmentation only with validated knobs: arrival jitter, lifetime perturbation, and topology/link failures.

Diagram to design:
- Roadmap with three lanes:
  - `Data`: lifetime-stratified sampling.
  - `Reward`: terminal/PBRS/optimal-gap shaping.
  - `System`: faster compiled training / hybrid online policy.

Professor-facing explanation:
- Future work should be framed as reducing deployment risk: identify workload regimes, choose policy accordingly, and keep a strong greedy fallback.

Sources:
- `docs/training/v4/reward-shaping.tex`: generalization gap and optimal-gap shaping.
- `docs/data_augmentation/data_aug.tex`: optional augmentation knobs.
- `docs/plans/v4/throughput_log.md`: system now supports faster experiments.

## Slides To Remove Or Fold

- [ ] Approved
- [ ] Needs edits

keep

Remove or fold:
- Timeline slides should be removed from the main narrative or moved to backup. They are useful project history but weaker than the result-first story. 
- MPD timeseries can be kept only if it directly supports the selected result. If it is from LON23 validation while the main result is LVL01/AMS20, put it in backup or label it clearly. 
- Detailed SAC internals should stay out of the main deck unless professor asks; keep the explanation at "entropy prevents premature collapse." 

## Backup Slides

- Existing timeline: useful if asked "what did you do over the semester?"
- SAC internals: actor/critic/replay/entropy.
- Full R1--R5 reward table.
- Full trace-lifetime table.
- Greedy parity/debugging notes only if asked about baseline validity.

## Approval Checklist

- [ ] Choose final result framing: newer nuanced result or older strong RL-vs-greedy result.
- [ ] Decide whether to mention PPO specifically or use "on-policy alternatives."
- [ ] Confirm whether the main deck should include 10 slides plus title, or be shorter.
- [ ] Confirm whether MPD timeseries remains in main deck or backup.
- [ ] Confirm if "optimal" should be presented as a lower bound only, or if there is a separate full-trace oracle result to include.
