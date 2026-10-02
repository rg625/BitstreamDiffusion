<!-- Plan as provided by the user on 2026-10-02 (pasted verbatim, pandoc math markup kept). -->

# CoBiT Adaptive Compute: Experimental Plan

## 0. Goal

Test whether **inference compute should be allocated according to the
current denoising state rather than uniformly across denoising steps**.

The immediate goal is **not** to design the final architecture.

The goal is to answer, with a small amount of compute:

> **Is there meaningful, predictable heterogeneity in the marginal value
> of additional compute across CoBiT denoising states, and at what
> granularity does that heterogeneity matter?**

If the answer is yes, the experiments should identify the simplest
useful form of adaptive computation before any MoE, routing, or
architectural modification is attempted.

------------------------------------------------------------------------

# 1. Core hypotheses

### H1 --- There is heterogeneous compute demand

At the same nominal denoising stage, different states may benefit by
very different amounts from another unit of inference compute.

Formally, for a state (x_t):

\[ Q(x_t, c) \]

is the final generation quality after allocating (c) additional units of
compute.

Define marginal value:

\[ M(x_t,c) = Q(x_t,c+`\Delta `{=tex}c)-Q(x_t,c) \]

The key question is whether:

\[ Var(M(x_t,c)`\mid `{=tex}t) \]

is substantial.

If it is not, state-dependent allocation is unlikely to be useful.

------------------------------------------------------------------------

### H2 --- Compute demand is not determined by timestep alone

Compare:

\[ M = f(t) \]

against:

\[ M = f(`\text{state}`{=tex}) \]

and:

\[ M = f(t,`\text{state}`{=tex}) \]

The interesting outcome is evidence that state information explains
additional variation beyond timestep.

------------------------------------------------------------------------

### H3 --- Simple state statistics may predict compute value

Candidate predictors:

-   token/bit entropy
-   mean and variance of confidence
-   fraction of uncertain bits/tokens
-   change in entropy from the previous step
-   change in predictions from the previous step
-   prediction disagreement / instability
-   hidden-state statistics
-   bitstream-specific statistics
-   trajectory-level convergence statistics

We should initially prefer **cheap, interpretable features**.

------------------------------------------------------------------------

### H4 --- The useful allocation granularity can be identified offline

Possible levels:

1.  **Example-level:** some sequences receive more total compute.
2.  **Temporal:** different denoising states receive different amounts
    of compute.
3.  **Token/bit-level:** different parts of a sequence receive different
    amounts of compute.

Do not assume the finest granularity is useful.

------------------------------------------------------------------------

# 2. Experimental philosophy

## Do first

-   Small-scale experiments
-   Offline counterfactual evaluation
-   Paired comparisons
-   Simple predictors
-   Explicit baselines
-   Fixed compute budgets
-   Final generation quality as the main outcome
-   Statistical confidence intervals

## Do not do yet

-   MoE architecture design
-   RL-based routing
-   Large-scale training
-   Complex learned controllers
-   Architecture search
-   Extensive hyperparameter sweeps
-   Token-level sparse execution

The principle is:

> **Measure the opportunity before building the mechanism.**

------------------------------------------------------------------------

# 3. Experiment 1 --- Does adaptive compute have an opportunity?

## Objective

Establish whether additional compute has highly variable value across
states at the same denoising stage.

This is the most important experiment.

If this experiment is negative, stop.

------------------------------------------------------------------------

## Dataset

Start with approximately:

-   **500--1,000 prompts**
-   representative of the intended evaluation distribution
-   fixed prompt set for all experiments

Use a small enough model/evaluation setup that the complete experiment
can run quickly.

Do not optimize for leaderboard-quality absolute scores yet.

------------------------------------------------------------------------

## Trajectory collection

For each prompt, run the normal CoBiT generation trajectory and save
approximately:

-   **6--8 denoising checkpoints**

At each checkpoint save enough information to reconstruct or evaluate
the state:

-   current bit/token state
-   timestep / noise level
-   model outputs or logits if available
-   confidence/entropy statistics
-   hidden-state summaries if inexpensive
-   random seed / sampling state
-   final generated sample

The exact saved representation should be chosen so that additional
analysis does not require rerunning the entire trajectory.

------------------------------------------------------------------------

# 4. Counterfactual compute branches

At each saved state, create branches with different amounts of
additional compute.

Initial choices:

``` text
+0
+1
+2
+4
```

If inexpensive, add:

``` text
+8
```

The exact meaning of "+1" must be defined carefully.

For the first study, use the simplest controlled notion of additional
compute available in the existing CoBiT inference procedure.

Avoid mixing several sources of extra compute in the same experiment.

------------------------------------------------------------------------

## Important methodological point

Do not casually interpret "one extra denoising step" as a causal unit of
compute unless the noise schedule makes the comparison well-defined.

For every branch, document:

-   number of model evaluations
-   noise/timestep schedule
-   whether the state is refined at the same noise level
-   whether the additional step changes the final noise trajectory
-   sampling randomness

The comparison must distinguish:

> **more compute**

from:

> **a different inference trajectory.**

If necessary, define compute in terms of **model evaluations / NFE**
first and defer wall-clock measurements.

------------------------------------------------------------------------

# 5. Evaluation metric

The primary quantity should be **final generation quality**.

For each state (x_t):

\[ Q_c(x_t) \]

= quality after (c) additional compute units.

Then calculate:

\[ `\Delta `{=tex}Q\_{0`\rightarrow1`{=tex}} = Q_1-Q_0 \]

\[ `\Delta `{=tex}Q\_{1`\rightarrow2`{=tex}} = Q_2-Q_1 \]

and so on.

Also calculate compute efficiency:

\[ `\eta `{=tex}= `\frac{\Delta Q}{\Delta C}`{=tex} \]

------------------------------------------------------------------------

## Evaluation strategy

Because final evaluation may be expensive:

### Stage A --- Cheap evaluation

Run the full trajectory dataset through inexpensive diagnostics such as:

-   entropy
-   confidence
-   prediction change
-   bit accuracy / denoising loss where meaningful
-   state statistics

### Stage B --- Expensive evaluation

Evaluate final generation quality on a smaller but representative
subset.

Use the same prompts and paired random seeds wherever possible.

This gives much more information per GPU-hour than evaluating every
branch with an expensive metric.

------------------------------------------------------------------------

# 6. The first decisive plot

Make this plot first:

> **Marginal benefit of additional compute vs. state, grouped by
> timestep.**

For example:

``` text
x-axis: state-level uncertainty / convergence statistic
y-axis: Q(+1) - Q(+0)
group/color: denoising timestep
```

Also make a simple distribution plot:

``` text
timestep
   |
   |   marginal benefit distribution
   |
   +------------------------------
```

The critical observation is **within-timestep variation**.

------------------------------------------------------------------------

# 7. Decision criterion after Experiment 1

### Case A --- Little variation

If marginal benefit is nearly constant within each timestep:

``` text
state -> compute
```

probably has little value.

Focus instead on a timestep-dependent schedule:

``` text
t -> compute
```

and stop the adaptive-state project.

------------------------------------------------------------------------

### Case B --- Strong within-timestep variation

If states at the same timestep have substantially different marginal
compute value:

Proceed.

This is the evidence that motivates state-dependent adaptive
computation.

------------------------------------------------------------------------

### Case C --- Large variation but no useful cheap predictor

The phenomenon may exist but be difficult to exploit.

Do not immediately build a large router.

Proceed to Experiment 2 to determine whether the relevant signal is
present in accessible state information.

------------------------------------------------------------------------

# 8. Experiment 2 --- What predicts the value of compute?

## Objective

Determine whether simple state features explain marginal compute value.

Fit very small predictors.

------------------------------------------------------------------------

## Models

Compare:

### Model A --- Timestep only

\[ M = f(t) \]

### Model B --- State statistics

\[ M = f(H) \]

where (H) represents uncertainty/confidence statistics.

### Model C --- State dynamics

\[ M = f(H,`\Delta `{=tex}H,`\Delta`{=tex}\^2 H) \]

### Model D --- Prediction dynamics

Add:

-   change in model predictions
-   prediction disagreement
-   confidence change

### Model E --- Bitstream features

Add CoBiT-specific bit-level statistics.

### Model F --- Full feature set

Combine the inexpensive features.

------------------------------------------------------------------------

# 9. Critical ablation: state-only vs timestep+state

This is one of the most informative tests.

Compare:

\[ f(`\text{state}`{=tex}) \]

against:

\[ f(t,`\text{state}`{=tex}) \]

Use cross-timestep validation.

For example:

-   train on some timesteps
-   test on held-out timesteps

The question is:

> Can the model recognize that a state needs more compute without
> explicitly being told its timestep?

If yes, this supports the hypothesis that compute demand is an intrinsic
property of the denoising state rather than simply a property of time.

If adding timestep provides a large improvement, timestep remains an
important component of the controller.

Neither result is a failure; they answer different scientific questions.

------------------------------------------------------------------------

# 10. What to measure

For each predictor report:

-   (R\^2) or rank correlation with marginal compute value
-   prediction error
-   cross-timestep generalization
-   calibration
-   inference cost of the predictor

The predictor does not need to perfectly predict quality.

The useful question is:

> **Can it reliably distinguish states where extra compute is valuable
> from states where it is not?**

------------------------------------------------------------------------

# 11. Experiment 3 --- At what granularity should compute be allocated?

Once heterogeneous compute demand is established, perform an **offline
oracle allocation experiment**.

Do not implement a router yet.

Assume a fixed total compute budget and ask how well an ideal allocator
could spend it.

Compare:

### A. Uniform allocation

Every state receives the same compute.

### B. Example-level allocation

Some complete sequences receive more compute.

### C. Temporal allocation

Different denoising states receive different compute.

\[ C_t \]

### D. Local allocation

Different tokens/bits receive different compute.

\[ C\_{t,i} \]

or

\[ C\_{t,i,b} \]

------------------------------------------------------------------------

# 12. Why the oracle experiment matters

This separates two questions:

### Question 1

Is adaptive computation theoretically useful?

### Question 2

Where does the useful granularity live?

If example-level allocation captures most of the achievable improvement,
there is no reason to build token-level routing.

If temporal allocation provides most of the gain, a simple adaptive
denoising scheduler may be sufficient.

Only if local allocation provides a meaningful additional gain should we
investigate token/bit-level conditional computation or MoE.

------------------------------------------------------------------------

# 13. Fixed-budget comparison

Compare methods at equal total compute.

Example budgets:

``` text
25% of baseline compute
50%
75%
100%
125%   (optional)
```

For each budget compare:

``` text
Uniform
Best timestep-only heuristic
State-based heuristic
Learned state predictor
Offline oracle
```

The oracle provides an approximate upper bound on what an adaptive
policy could achieve using the available state information.

------------------------------------------------------------------------

# 14. Recommended plots

The minimum useful figure set is:

## Figure 1 --- Heterogeneity

Marginal value of extra compute vs state, grouped by timestep.

## Figure 2 --- Predictor quality

Prediction of marginal compute value for:

``` text
timestep
state
state + dynamics
state + timestep
```

## Figure 3 --- Oracle allocation

Final quality vs total compute for:

``` text
uniform
example-level oracle
temporal oracle
local oracle
```

## Figure 4 --- Distribution of compute demand

For a fixed total compute budget, show how much compute the oracle
assigns to different states/timesteps.

These four figures should be enough to decide whether the project is
worth pursuing and what direction it should take.

------------------------------------------------------------------------

# 15. Statistical design

Use paired evaluation wherever possible.

For each prompt:

``` text
same prompt
same initial state
same random seed
different compute allocation
```

This reduces variance substantially.

Report:

-   mean paired improvement
-   bootstrap confidence interval
-   distribution across prompts

Avoid relying only on aggregate means.

A useful result should show that the adaptive advantage is not driven by
a small number of outlier prompts.

------------------------------------------------------------------------

# 16. Compute-efficient execution strategy

Use a funnel.

### Phase 1 --- Very cheap

``` text
100–200 prompts
6–8 checkpoints
simple state statistics
small number of branches
```

Goal:

> Is there obvious heterogeneity?

If no → stop.

------------------------------------------------------------------------

### Phase 2 --- Moderate

``` text
500–1,000 prompts
6–8 checkpoints
+0/+1/+2/+4 branches
```

Goal:

> Quantify heterogeneity and identify predictors.

------------------------------------------------------------------------

### Phase 3 --- Targeted expensive evaluation

Select a stratified subset covering:

-   low predicted compute value
-   medium predicted compute value
-   high predicted compute value
-   different timesteps

Evaluate final generation quality carefully.

Goal:

> Confirm that the observed signal actually corresponds to useful
> generation improvement.

------------------------------------------------------------------------

# 17. Do not start with MoE

MoE should be treated as a **mechanism**, not the hypothesis.

First establish:

``` text
heterogeneous compute demand
        ↓
predictable from state
        ↓
useful at a particular granularity
        ↓
adaptive allocation produces a real quality/compute gain
        ↓
choose implementation
```

Only then ask whether MoE is an appropriate implementation.

Possible mechanisms later include:

-   adaptive number of denoising steps
-   repeated refinement
-   conditional depth
-   conditional FFNs
-   sparse token/bit processing
-   compute-specialized experts
-   mixture of shallow/deep refinement paths

------------------------------------------------------------------------

# 18. Suggested minimal first run

If only one experiment can be run, do this:

``` text
1,000 prompts
×
8 denoising checkpoints
×
{+0, +1, +2, +4 compute}
```

For every checkpoint record:

``` text
state
timestep
entropy/confidence
prediction change
bit-level statistics
random seed
```

Then compute:

\[ `\Delta `{=tex}Q = Q(+1)-Q(+0) \]

and plot its distribution **within each timestep**.

This single experiment answers the most important first question:

> **At a fixed denoising stage, do different CoBiT states have
> meaningfully different marginal values of additional compute?**

If the answer is no, stop.

If yes, continue with feature prediction and oracle allocation.

------------------------------------------------------------------------

# 19. Final decision tree

``` text
                    Start
                      |
                      v
          Measure marginal compute value
                      |
          +-----------+-----------+
          |                       |
     little variation        large variation
          |                       |
          v                       v
   timestep schedule       Can state features
       may suffice           predict value?
                                  |
                         +--------+--------+
                         |                 |
                        no                yes
                         |                 |
                         v                 v
                investigate richer    Determine useful
                trajectory signals     allocation granularity
                                           |
                              +------------+------------+
                              |            |             |
                           example      temporal       local
                              |            |             |
                              v            v             v
                           adaptive    adaptive      conditional
                           stopping    scheduler     local compute
                                                        |
                                                        v
                                                  only now consider
                                                  MoE / sparse routing
```

------------------------------------------------------------------------

# 20. Success criteria

The project should only move to architecture work if the experiments
establish all three:

### 1. Opportunity

There is substantial variation in marginal compute value between states.

### 2. Predictability

That variation can be predicted from information available cheaply
during inference.

### 3. Exploitability

An offline adaptive allocation strategy produces a meaningful quality
improvement at fixed compute, or equivalent compute savings at fixed
quality.

If any of these three fail, revise or abandon the adaptive-compute
direction rather than increasing architectural complexity.

------------------------------------------------------------------------

# 21. Immediate next actions

### Step 1

Implement trajectory/checkpoint saving.

### Step 2

Implement deterministic counterfactual branching from saved states.

### Step 3

Run the small pilot:

``` text
100–200 prompts
×
6–8 checkpoints
×
+0/+1/+2/+4
```

### Step 4

Produce Figure 1:

> marginal value of additional compute vs state, grouped by timestep.

### Step 5

Inspect within-timestep variance.

### Step 6

If promising, scale to 500--1,000 prompts.

### Step 7

Fit the small state-only predictors.

### Step 8

Run the offline oracle allocation experiment.

### Step 9

Only after these results decide whether an adaptive inference mechanism
is worth implementing.

------------------------------------------------------------------------

# Bottom line

The project should initially be framed as a **measurement and allocation
problem**, not an architecture problem.

The first scientific question is not:

> "What MoE should we build?"

It is:

> **"Where, in the CoBiT denoising trajectory, does one additional unit
> of compute have the highest marginal value, and can that value be
> predicted from the current state?"**

A small counterfactual dataset, a few simple predictors, and an offline
allocation oracle should provide most of the evidence needed to answer
that question before committing substantial engineering or training
compute.
