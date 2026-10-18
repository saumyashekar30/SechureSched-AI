# PPO scheduling component

Everything here is **additive**: `main.py`, the existing schedulers, `risk_model.py`, the simulator and
`results/raw_results.csv` are unchanged. The PPO code lives in `src/ppo/` and needs `requirements-ppo.txt`;
the original pipeline still runs without those packages.

## Files

| File | Purpose |
|---|---|
| `src/ppo/sched_env.py` | Gymnasium environment (non-preemptive single-CPU scheduler) |
| `src/ppo/config.py` | `EnvConfig` (reward weights, K) and `TrainConfig` (PPO hyper-parameters, data split, preserved IF/λ settings) |
| `src/ppo/workloads.py` | Extra workloads from the same Alibaba trace, train/val/test split, vectorised risk scoring |
| `src/ppo/train.py` | MaskablePPO training, reward log, validation-based best checkpoint, config + risk model saved |
| `src/ppo/inference.py` | `PPOScheduler.load(...).schedule(df)` – deterministic inference; same output schema as `run_simulation` |
| `src/ppo/evaluate.py` | Experiments A/B, tables, plots |
| `src/evaluation/security_metrics.py` | Detector precision/recall/F1 and scheduler-level security violations |
| `tests/test_ppo.py` | 12 tests |

## MDP

**Scheduling model.** Single CPU, *non-preemptive*. A decision is taken whenever the CPU becomes free and
the ready queue is non-empty; the chosen process then runs for its full actual burst. If the queue is empty the
clock jumps to the next arrival. This is exactly the model of the existing FCFS/SJF/SecureSched code: with
the candidate window made large enough, the environment reproduces `run_simulation` FCFS and SJF metrics
bit-for-bit (tested, and identical to the committed `raw_results.csv` values). Round Robin is preemptive and is
only a baseline.

**Observation** (69 floats in [0,1]): for each of K=16 candidate slots `[valid, declared_burst, waiting_time, risk]`,
plus 5 global features `[queue length, share of queue not shown, mean queue risk, share of queue flagged (risk>0.5), CPU utilisation so far]`.
Times are divided by the running median declared burst of the processes that have arrived so far
(floor 0.5) and log-squashed (`log1p(x)/log1p(100)`, clipped to 1) because trace bursts are heavy-tailed.
The risk is the existing Isolation-Forest score. The agent does **not** see the actual burst, the future,
`is_malicious`, or the reward normaliser (a test flips every label and checks observations and rewards are identical).
"Priority" does not exist in the dataset, so it is not used; "execution state" is the single-CPU busy
fraction.

**Action** `Discrete(K)`: index of a candidate slot. If more than K processes are ready, the window holds the K/4
oldest (anti-starvation), the K/2 shortest-declared among unflagged, and the shortest-declared of the rest.
`MaskablePPO` (sb3-contrib) masks empty slots, so invalid actions cannot be sampled; if an invalid index is
passed anyway, `step` falls back to the oldest candidate, counts it and penalises it.

**Transition.** Deterministic given the workload: run the chosen process to completion, admit arrivals,
jump over idle gaps. An episode ends when all processes are complete (one decision per process).

**Reward** (per decision, `p` = chosen process, `u` = reward unit, see below; all costs clipped at 20):

```
r = - [ w_wait * Δwait/u  +  w_fair * Δaged/u  +  w_sec * R_p * n_safe/n_w  +  w_idle * min(idle/ref,10)/10 ]
```

* `Δwait` = exact extra waiting time the dispatch inflicts on everyone else (`b·|others|` plus the overlap of
  processes that arrive while `p` runs). Σ Δwait over an episode equals Σ waiting time (tested). Because
  Σ turnaround = Σ waiting + Σ burst and the latter is constant, this single term also minimises mean turnaround.
* `Δaged` = same delay but each waiting process weighted by `min(1, wait/(τ·ref))`, τ=50 → delaying processes that
  already waited long costs more (starvation/fairness).
* security term: `R_p` is the risk of the chosen process, `n_safe/n_w` the share of *other* waiting processes that
  are unflagged; applied only if `R_p > 0.5`. It penalises dispatching a flagged process ahead of unflagged ones but is
  zero when only flagged processes remain, so it never pays to hold suspicious processes back forever (they also keep
  accruing waiting cost).
* idle term: the policy is work-conserving, so idle time is fixed by arrivals; the term is kept for
  completeness and is constant across policies.
* `u = max(mean FCFS waiting per process of the episode, median burst, 1)` only rescales rewards across episodes of
  very different scale. It uses simulator ground truth, but only inside the reward, never in the observation.
* Weights: `w_wait=1, w_fair=0.25, w_sec=1, w_idle=0.1`. They were set once by judgement and **not tuned**.
  Trade-off: a larger `w_sec` pushes towards SecureSched-like behaviour (fewer violations, possibly more waiting
  for false positives, which are common: ~16% of benign held-out processes are flagged).

## Why PPO, and architecture

On-policy, stable with a discrete action space and cheap simulator rollouts; action masking handles the variable queue.
Network: separate actor and critic MLPs `69 → 128 → 128 → (16 | 1)`, tanh (SB3 default). Hyper-parameters (`TrainConfig`):
150 000 timesteps, lr 3e-4, `n_steps` 1024, batch 128, γ 0.99, clip 0.2, GAE λ 0.95, entropy 0.01, 10 epochs, CPU.
Training takes ~6 min per seed on 1 core.

## Data and methodology

* Existing, preserved: seed 42, IF `contamination=0.1, random_state=42`, λ=500, attack ratios 0/5/10/15/20 %,
  `inject_anomalies` unchanged.
* The committed `data/processed/processes.csv` contains only **20 near-identical processes** (bursts 53–62, all
  arriving within 5 s) – too small to train on. Additional workloads: from the first 50 000 rows of
  `batch_instance.csv` (same cleaning as the preprocessor), the sorted trace is cut into 188 disjoint blocks of
  250 consecutive instances; 50 instances are sampled from each block as one workload window. Syscall rates use
  the same N(30,5) clipped distribution (synthetic, as in the existing pipeline). Windows are randomly (seed 42)
  split into 120 train / 15 validation / 30 test. The Isolation Forest for these experiments is fitted on clean
  training windows only. Test windows are never used for training, model selection or the IF fit.
* Training episodes: train windows × the five attack ratios (random length 20–50). The best checkpoint is
  chosen on the validation windows. 3 training seeds (42, 43, 44).
* **Experiment A** – committed workload, original seed-42 protocol (IF fitted on the clean workload itself).
  The baselines are checked against `results/raw_results.csv` and match exactly. The PPO policy trained on trace windows is applied to it
  without retraining, so this is also a distribution-shift test (its risk scores come from a different IF).
* **Experiment B** – 30 held-out windows × 5 ratios × 3 injection seeds (42,43,44) = 90 workloads per ratio,
  all policies on identical workloads. Reported: mean and std over workloads (the std is dominated by
  heavy-tailed workload differences, so paired differences are also provided).

### Security metrics
* Precision / recall / F1: Isolation Forest flag (`risk>0.5`) vs injected labels. Property of the detector,
  identical for all schedulers; undefined values (no attackers at 0 %) are NaN.
* Security violation (labels used for evaluation only): a malicious process commits a *harmful priority inversion*
  if a benign process that had already arrived when it first got the CPU, with strictly shorter actual burst,
  completes after it. `violation_rate` = share of malicious processes that do. Running an attacker is not itself a
  violation. FCFS/SJF violate by construction; FCFS is "violating" without any gaming (it ignores burst).
* CPU utilisation/throughput are not reported: all policies are work-conserving, so they are identical.

## Reproduce

```bash
pip install -r requirements-ppo.txt
python -m pytest tests -q
for s in 42 43 44; do python -m src.ppo.train --seed $s --out-dir results/ppo/models/seed_$s; done
python -m src.ppo.evaluate            # ~5 min; writes results/ppo/*.csv, *.png
```
Options: `--timesteps --learning-rate --n-steps --batch-size --gamma --clip-range`.

## Measured results (from the committed `results/ppo/*.csv`)

Held-out windows, mean avg. waiting time (time units), PPO = mean of 3 training seeds:

| attack % | FCFS | SJF | RR | SecureSched | PPO |
|---|---|---|---|---|---|
| 0 | 838 | 778 | 1527 | 796 | 809 |
| 5 | 995 | 1102 | 1547 | 903 | 928 |
| 10 | 1276 | 1581 | 1618 | 1089 | 1159 |
| 15 | 1437 | 1886 | 1696 | 1209 | 1322 |
| 20 | 1674 | 2296 | 1861 | 1409 | 1576 |

Violation rate (20 %): FCFS 0.98, SJF 1.00, RR 0.00, SecureSched 0.17, PPO 0.41. Detector F1 at 20 %: 0.86 (precision 0.81, recall 1.0).

**Conclusions supported by the measurements**
* PPO clearly beats SJF and FCFS under attack and RR on waiting time (paired difference vs SJF is negative in
  ≥ 98 % of workloads for 5–20 %).
* PPO does **not** beat the existing heuristic SecureSched: its mean waiting time is higher at every ratio
  (paired mean difference +13 … +167, about 1.6–2.5 standard errors from 0, so suggestive but not conclusive), and its violation rate is higher
  (0.30–0.41 vs 0.17–0.20). At low ratios PPO is better than SecureSched in 60–76 % of individual workloads but worse on average
  (heavy-tailed losses).
* Across PPO training seeds the 20 % result varies from 1432 to 1659 (`expB_ppo_by_training_seed.csv`).
* On the 20-process committed workload (Exp. A) PPO is close to SecureSched (e.g. 657 vs 623 at 20 %), with
  violations > 0 at 10–20 %.

## Limitations
* The reward curves were still rising at 150 k steps (`training_reward_curve.png`); longer training, reward-weight tuning
  (e.g. larger `w_sec`) or imitation warm-start were **not** tried, so the gap to SecureSched may not be fundamental.
* Non-preemptive model only; true preemptive PPO scheduling is future work.
* Attackers are the synthetic dual-signature ones (detector recall is 1.0 by construction); there is no adaptive
  attacker. Syscall rates are synthetic.
* Window split is random across one trace prefix, so test windows are temporally interleaved with training windows
  (same distribution); generalisation to other traces/clusters is untested.
* Workloads are near-simultaneous batches (windows span ~1 s of arrivals), where scheduling is essentially ordering.
* Benign false positives (~16 % of held-out benign processes flagged) cost both SecureSched and PPO.
