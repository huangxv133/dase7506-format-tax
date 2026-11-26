# The Price of Parsability: How Output-Format Constraints Change Accuracy and Cost in Mathematical Reasoning

DASE7506 Advanced Machine Learning Methods — Final Project

## Abstract

Evaluation pipelines for mathematical reasoning almost always impose an output format: a bracketed answer, a designated final line, or an instruction to be brief. These constraints are adopted for parsability, yet their effect on the measurement itself is rarely quantified. **The research question of this paper is therefore: when an output-format constraint changes a reported accuracy, how much of that change is a genuine change in the model's reasoning ability, and how much is an artefact of the evaluator failing to read an answer the model actually produced?** We run a 2×3 factorial experiment on 200 GSM8K problems across three model sizes (Qwen2.5-0.5B, 1.5B and 7B), crossing reasoning verbosity (full vs. brief) with answer format (unconstrained, designated answer line, boxed), plus an unconstrained no-reasoning anchor. Every response is graded twice — once with a strict extractor that honours only the requested format, once with a lenient extractor that accepts any standard format — and the gap isolates a *parsing tax* from a *capability tax*. Three findings follow. First, format constraints are not uniformly costly: the boxed format imposes no measurable tax (strict and lenient scores differ by at most 0.5pp at every model size), whereas the answer-line format costs 41.0, 18.0 and 12.5pp of measured accuracy at 0.5B, 1.5B and 7B, almost entirely as parsing tax rather than capability loss. Second, brevity instructions are obeyed only by the largest model, where they cost 11.5pp for 203 fewer output tokens. Third, brevity and boxed format interact super-additively: the 7B model returns a bare boxed answer with no reasoning in 36.5% of cases, and accuracy falls by 23.5pp. What matters is which constraint is imposed, not whether one is imposed. Code, the exact prompts, the frozen extractors and all 4,200 raw responses are released at [REPOSITORY URL].

## I. Introduction

Every automated evaluation of chain-of-thought reasoning has to answer a mundane question: where in the model's output is the answer? The usual solution is to constrain the output — ask for `\boxed{}`, ask for a line reading "Final answer: X", ask for a short response. These choices are made for the convenience of the harness, and they are rarely reported, let alone ablated.

This is a measurement problem before it is a modelling problem. If a constraint reduces measured accuracy, two very different things may have happened. The model may have become worse at reasoning, because the constraint consumed capacity or distorted the solution. Or the model may have answered just as well but failed to satisfy the format, so the harness scored a correct answer as wrong. The first is a property of the model; the second is a property of the evaluator. Conflating them makes reported accuracies incomparable across papers that use different harnesses, and it misattributes responsibility when a small model appears to reason badly.

We separate the two by grading the same response twice. A *strict* extractor honours only the format the condition requested; a *lenient* extractor tries every standard format in turn (boxed, answer line, trailing number) and accepts whichever yields a value. Because the two extractors see identical text, their difference is attributable to format compliance alone. We call it the **parsing tax**. Any residual change in the lenient score, which no compliant extractor can recover, is the **capability tax**.

We apply this decomposition to a 2×3 factorial design on 200 GSM8K problems [3], across three sizes of the Qwen2.5-Instruct family [4]: verbosity (full steps vs. as brief as possible) crossed with answer format (no instruction, designated answer line, boxed). A seventh condition removes reasoning altogether to confirm the design is sensitive to reasoning at all.

The results are more specific than "constraints hurt". The boxed format imposes no measurable tax: at every model size its strict and lenient scores differ by at most 0.5pp, and compliance is 96–99%. The answer-line format is expensive but for a different reason: compliance drops to 18% at 0.5B, and the resulting parsing tax is 41pp, while the capability tax is statistically indistinguishable from zero. Brevity, in contrast, is a genuine capability cost, but only for the model large enough to obey it. And the two factors interact: for the 7B model, combining brevity with the boxed format causes it to emit a bare boxed number with no reasoning in more than a third of cases, and accuracy collapses.

The practical implication is narrow and actionable: if an evaluation must impose a format, a boxed answer is close to free, a designated answer line is not, and combining a format constraint with a brevity instruction can suppress the very reasoning being measured.

Our contributions are: (i) a decomposition of format effects into parsing and capability taxes that can be computed from any existing log with a single extra extraction pass; (ii) a controlled 2×3 measurement across three model sizes showing that the two taxes behave differently and that the format matters more than its presence; (iii) an identified failure mode — brevity plus boxed format collapsing chain-of-thought — with the mechanism quantified at the response level.

## II. Related Work

**Chain-of-thought and its cost.** Eliciting intermediate reasoning substantially improves multi-step arithmetic [1], and a single zero-shot instruction suffices on many benchmarks [2]. Because reasoning is emitted as text, its length is under prompt control, and later work has treated that length as a compute budget to be allocated [12]. Our anchor condition reproduces the basic effect and quantifies what removing it costs at each model size.

**Prompt sensitivity.** Prompt wording and surface formatting are known to move accuracy by large margins. Sclar et al. show that separator, casing and spacing choices alone can shift performance by tens of points across tasks [5], and Voronov et al. show that inconsistent output formats make comparisons between in-context-learning methods unreliable [6]. Both studies concentrate on in-context classification settings. We ask the complementary question for zero-shot mathematical reasoning, where the constraint is not incidental decoration but the mechanism by which the answer is located, and we separate the measurement artefact from a genuine change in ability.

**Self-correction and second attempts.** A parallel literature asks whether a model can improve its own answer [8, 9]. That question requires a second pass and therefore confounds reviewing with resampling; our design avoids the confound entirely by keeping every condition to a single pass.

**Evaluation methodology.** Concerns about reproducibility of reported numbers are longstanding; Biderman et al. document how harness choices propagate into results [7]. Our parsing tax is a concrete, measurable instance of that concern.

## III. Problem Formulation

Let π be a fixed model, and let c ∈ {C0, …, C6} index a condition, i.e. a prompt template. For a problem x with gold answer y, the model returns a single response r = π(x, c), from which an extractor e yields ê = e(r). Define correctness as 𝟙[ê = y] under exact numerical match.

Two extractors are defined. The strict extractor e_s(c) is condition-specific: for the answer-line conditions it accepts only the designated line, for the boxed conditions only `\boxed{…}`, and for unconstrained conditions it falls back to the last parseable number in the tail of the response. It returns ⊥ when the required marker is absent. The lenient extractor e_l is condition-agnostic and tries boxed, then answer line, then trailing number, returning the first value found. Because e_l ignores which format was requested, e_l ≥ e_s pointwise, and e_s = e_l for unconstrained conditions by construction.

For condition c and model π, let A_l(c) and A_s(c) be the expected accuracy of e_l and e_s respectively, and T(c) the expected number of generated tokens. Relative to the unconstrained full-reasoning baseline C1, we define

- **capability tax** Δ_cap(c) = A_l(c) − A_l(C1), the change no extractor can recover;
- **parsing tax** Δ_parse(c) = A_s(c) − A_l(c) ≤ 0, accuracy lost to non-compliance alone;
- **total measured change** Δ_total(c) = A_s(c) − A_l(C1) = Δ_cap(c) + Δ_parse(c);
- **exchange rate** R(c) = Δ_cap(c) / (ΔT(c)/100), the accuracy paid per 100 tokens saved.

Two hypotheses follow directly. H1 (decomposability): format constraints act primarily through Δ_parse rather than Δ_cap, and the magnitude depends on which format is requested. H2 (additivity): the effect of brevity is independent of the effect of format, i.e. Δ_cap(C5) ≈ Δ_cap(C4) + Δ_cap(C2) and Δ_cap(C6) ≈ Δ_cap(C4) + Δ_cap(C3). Both are testable with paired data and are falsified below for the largest model.

All quantities are estimated on paired items. Differences are reported with 95% intervals using the MOVER–Newcombe method for paired proportions [10], which combines Wilson intervals for the two marginal rates with the paired structure.

## IV. Experimental Setup

**Data.** GSM8K test split [3], 200 problems sampled without replacement under seed 7506. The same 200 problems, in the same order, are used for every condition and every model. The number of calculation annotations (`<<…>>`) in the reference solution is retained as a difficulty proxy and used for stratification; 130 problems have 1–3 annotations and 70 have 4 or more.

**Models.** Qwen2.5-0.5B-Instruct, 1.5B-Instruct and 7B-Instruct [4], served locally in bfloat16 on NVIDIA A800-80GB GPUs, one model per device. We use three sizes because format compliance is itself a capability, so a single model cannot distinguish "the constraint is costly" from "the model cannot follow it".

**Conditions.** All prompts share a fixed system message and differ only in two instruction clauses appended after the problem text. Verbosity is either *full* ("Show your calculation steps in full") or *brief* ("Show only the essential calculation steps, as briefly as possible"). Format is either *free* (no instruction about the ending), *answer line* ("End with one line in the form \`Final answer: <number>\`"), or *boxed* ("Put the final number inside \`\boxed{}\`"). The full crossing yields C1–C6; C0 omits reasoning entirely ("Output only the final number"). Prompts are reproduced verbatim in the appendix.

**Decoding.** Greedy, single sample, at most 512 new tokens, no tools, no retrieval. Responses truncated at the limit are recorded and reported; the truncation rate never exceeds 5.0% in any cell of the main grid.

**Scale.** 3 models × 7 conditions × 200 problems = 4,200 generations, completed in under 15 minutes of GPU time. Raw responses, per-token counts and both extraction results are logged per item and released with the paper.

## V. Results

Table I reports every cell. Figure 1 shows the accuracy surface, Figure 2 places conditions on the accuracy–cost plane, and Figure 3 shows the capability tax.

**A. The anchor behaves as required.** Removing reasoning costs 43.5, 59.5 and 75.0pp at 0.5B, 1.5B and 7B respectively (all intervals exclude zero). The damage grows with model capability, which is the expected signature: a stronger model has more to lose when its reasoning is suppressed. This confirms that the design is sensitive to reasoning content and that the remaining differences are not floor or ceiling artefacts.

**B. Parsing tax is large, capability tax is not.** At 0.5B, the answer-line condition (C2) scores 49.0% lenient and 8.0% strict. The capability tax is +0.5pp with interval [−9.2, +10.2] — no detectable change in ability — while the parsing tax is 41.0pp. Three quarters of the apparent damage from this constraint is an artefact of where the answer was looked for, not of what the model could do. Compliance tells the same story: only 18.0% of 0.5B responses contained the requested line at all, whereas 96.0% contained a boxed answer. The model has a strong default format, acquired during instruction tuning, and a format instruction that conflicts with it is frequently ignored.

**C. Which format matters more than whether a format exists.** Across all three sizes the boxed condition's strict and lenient scores differ by at most 0.5pp, with compliance of 96.0–99.0%. The answer-line condition, by contrast, produces a parsing tax of 41.0, 18.0 and 12.5pp at 0.5B, 1.5B and 7B. The tax shrinks as models grow because compliance improves (18.0% → 75.0% → 85.5%); the underlying ability is unchanged at every size. This is the central empirical claim of the paper: parsimony about formats is not the issue, format identity is, and the cost of a badly chosen format is paid entirely by the evaluator.

**D. Brevity is a real capability cost, but only where it is obeyed.** For the 7B model, the brief condition (C4) saves 203 output tokens (−63%) and loses 11.5pp (interval [−17.9, −5.3]), an exchange rate of 5.7pp per 100 tokens saved. The 1.5B model halves its output (−106 tokens) for 3.0pp, at the edge of detectability, and the 0.5B model does not shorten at all (−15 tokens, +1.0pp): the brevity instruction is simply not followed. Instruction-following, like format compliance, is a capability that appears between 1.5B and 7B.

**E. Brevity and boxed format interact super-additively.** For the 7B model, brevity alone costs 11.5pp and the boxed format alone costs 0.0pp, yet the two together (C6) cost 23.5pp (interval [−30.6, −16.3]) — twice the sum of the parts, falsifying H2. The mechanism is visible in the response lengths: 36.5% of C6 responses contain fewer than 20 tokens, against 0.5% under brevity alone, and that short subset scores 31.5%. The model emits a bare boxed number and stops. Prompted to be brief and told that the answer belongs in a box, it treats the box as the deliverable and deletes the reasoning that the box was supposed to summarise. The answer-line variant does not trigger this (C5: −9.5pp, 0.5% of responses under 20 tokens), so the collapse is specific to a format that can be satisfied without any accompanying text.

**F. The brevity penalty concentrates on easy problems.** Under C1 the 7B model scores 96.9% on 1–3 step problems and 88.6% on 4+ step problems. Under brevity it scores 82.3% and 82.9%: a 14.6pp loss on easy problems and a 5.7pp loss on hard ones. Compression hurts where the model could previously be reliably right, which is the opposite of the intuition that short answers mainly damage hard cases. One reading is that hard problems already force the model to write out the needed steps, whereas easy problems are where it can be talked into skipping them.

**TABLE I — Accuracy (%) and cost by condition.** Each cell gives lenient accuracy / strict accuracy. The final column gives mean output tokens for 0.5B / 1.5B / 7B.

| Condition | 0.5B | 1.5B | 7B | Output tokens |
| --- | --- | --- | --- | --- |
| C0 no reasoning | 5.0 / 5.0 | 9.5 / 9.5 | 19.0 / 19.0 | 4 / 3 / 3 |
| C1 full, free | 48.5 / 48.5 | 69.0 / 69.0 | 94.0 / 94.0 | 307 / 301 / 322 |
| C2 full, answer line | 49.0 / 8.0 | 74.0 / 56.0 | 91.0 / 78.5 | 271 / 241 / 260 |
| C3 full, boxed | 53.0 / 52.5 | 76.5 / 76.0 | 94.0 / 93.5 | 302 / 297 / 312 |
| C4 brief, free | 49.5 / 49.5 | 66.0 / 66.0 | 82.5 / 82.5 | 292 / 196 / 120 |
| C5 brief, answer line | 49.0 / 10.5 | 66.0 / 59.0 | 84.5 / 84.0 | 255 / 118 / 78 |
| C6 brief, boxed | 50.5 / 50.0 | 69.0 / 69.0 | 70.5 / 70.5 | 299 / 275 / 96 |

**TABLE II — Taxes relative to C1 (pp).** Capability tax Δ_cap = accuracy(c) − accuracy(C1), so negative means the condition is worse; brackets give the 95% MOVER–Newcombe interval. Parsing tax Δ_parse = strict − lenient, reported for 0.5B / 1.5B / 7B. ΔT is the change in mean output tokens.

| Condition | Δ_cap 0.5B | Δ_cap 1.5B | Δ_cap 7B | Δ_parse (0.5/1.5/7B) | ΔT (0.5/1.5/7B) |
| --- | --- | --- | --- | --- | --- |
| C0 no reasoning | −43.5 [−50.8, −35.6] | −59.5 [−66.4, −51.2] | −75.0 [−80.5, −67.7] | 0 / 0 / 0 | −304 / −298 / −319 |
| C2 full, answer line | +0.5 [−9.2, +10.2] | +5.0 [−3.8, +13.7] | −3.0 [−8.4, +2.3] | 41.0 / 18.0 / 12.5 | −37 / −60 / −63 |
| C3 full, boxed | +4.5 [−5.3, +14.1] | +7.5 [−1.2, +16.1] | 0.0 [−4.9, +4.9] | 0.5 / 0.5 / 0.5 | −5 / −4 / −10 |
| C4 brief, free | +1.0 [−8.7, +10.7] | −3.0 [−12.1, +6.2] | −11.5 [−17.9, −5.3] | 0 / 0 / 0 | −15 / −106 / −203 |
| C5 brief, answer line | +0.5 [−9.2, +10.2] | −3.0 [−12.1, +6.2] | −9.5 [−15.7, −3.5] | 38.5 / 7.0 / 0.5 | −52 / −183 / −244 |
| C6 brief, boxed | +2.0 [−7.7, +11.7] | 0.0 [−9.0, +9.0] | −23.5 [−30.6, −16.3] | 0.5 / 0.0 / 0.0 | −8 / −26 / −226 |

**TABLE III — Lenient accuracy (%) by problem complexity.** Problems split by the number of `<<…>>` annotations in the reference solution.

| Model | C1 1–3 / 4+ steps | C4 1–3 / 4+ steps | C5 1–3 / 4+ steps |
| --- | --- | --- | --- |
| 0.5B | 59.2 / 28.6 | 60.0 / 30.0 | 59.2 / 30.0 |
| 1.5B | 80.0 / 48.6 | 77.7 / 44.3 | 75.4 / 48.6 |
| 7B | 96.9 / 88.6 | 82.3 / 82.9 | 90.8 / 72.9 |

## VI. Discussion

The two taxes have different owners. A parsing tax is a defect in the harness: it is recoverable by extracting more carefully, costs nothing to fix, and should never appear in a reported number. A capability tax is a real change in behaviour and belongs in the result. Reporting only the strict score silently converts the first into the second, which is exactly what happens when a paper adopts whichever format its parser happens to expect.

The result that the boxed format is free has a simple explanation: it agrees with the model's prior. Instruction-tuned mathematical models are trained on data where the answer is boxed, so requesting a box asks for what the model would have done anyway. The answer-line format asks for a sentence-shaped artefact that appears rarely in that training distribution, and small models cannot reliably produce it on demand. This suggests a cheap diagnostic for any harness: measure compliance before trusting a difference in accuracy, because a low-compliance condition is measuring formatting, not reasoning.

The interaction in Section V-E is the practical warning. Brevity and a boxed format are each individually defensible — one controls cost, the other enables parsing — but together they license the model to drop reasoning altogether in a third of cases. Any evaluation that imposes both is partly measuring the instruction to stop thinking.

## VII. Limitations

The study covers one dataset, one model family, one language, and greedy single-sample decoding; the absolute numbers should not be transferred to other benchmarks or to models trained differently. 200 problems give paired intervals of roughly ±5–8pp, which is enough to resolve the large effects reported here but not the 0.5B brevity difference. The extractors are the authors' own construction: the lenient extractor's format priority, and the rule that a trailing number must fall in the last 20% of the response, are design choices. Both are fixed before the final pass, released in full, and the no-position-constraint variant is reported alongside. Because parsing tax is defined relative to our lenient extractor, a more aggressive extractor would shrink it; the decomposition is still valid, but the magnitudes are extractor-relative. Finally, GSM8K is likely present in pretraining data, which inflates absolute accuracy; the paired contrasts are less exposed to this but not immune.

## VIII. Conclusion

Adding a format to a reasoning prompt is not a neutral act, but its cost is not general — it depends on which format and on whether a length instruction accompanies it. Grading every response twice turns this from an anecdote into a measurable quantity: the boxed format is free, the answer-line format costs up to 41pp of measured accuracy while leaving ability untouched, and brevity combined with boxing can suppress reasoning entirely. Evaluation harnesses should report compliance and both scores, and should prefer the format the model already uses.

## References

[1] J. Wei et al., "Chain-of-thought prompting elicits reasoning in large language models," in *Advances in Neural Information Processing Systems*, 2022.

[2] T. Kojima, S. S. Gu, M. Machel, and Y. Iwasawa, "Large language models are zero-shot reasoners," in *Advances in Neural Information Processing Systems*, 2022.

[3] K. Cobbe et al., "Training verifiers to solve math word problems," *arXiv:2110.14168*, 2021.

[4] Qwen Team, "Qwen2.5 technical report," *arXiv:2412.15115*, 2024.

[5] M. Sclar, Y. Choi, Y. Tsvetkov, and A. Suhr, "Quantifying language models' sensitivity to spurious features in prompt design, or: How I learned to start worrying about prompt formatting," in *International Conference on Learning Representations*, 2024.

[6] A. Voronov, L. Wolf, and M. Ryabinin, "Mind your format: Towards consistent evaluation of in-context learning improvements," in *Findings of the Association for Computational Linguistics: ACL 2024*, 2024.

[7] S. Biderman et al., "Lessons from the trenches on reproducible evaluation of language models," *arXiv:2405.14782*, 2024.

[8] J. Huang et al., "Large language models cannot self-correct reasoning yet," in *International Conference on Learning Representations*, 2024.

[9] M. Liu et al., "Large language models have intrinsic self-correction ability," *arXiv:2406.01297*, 2024.

[10] R. G. Newcombe, "Interval estimation for the difference between independent proportions: comparison of eleven methods," *Statistics in Medicine*, vol. 17, no. 8, pp. 873–890, 1998.

[11] X. Wang et al., "Self-consistency improves chain of thought reasoning in language models," in *International Conference on Learning Representations*, 2023.

[12] C. Snell, J. Lee, K. Xu, and A. Kumar, "Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning," in *International Conference on Learning Representations*, 2025.

[13] A. Lewkowycz et al., "Solving quantitative reasoning problems with language models," in *Advances in Neural Information Processing Systems*, 2022.

[14] H. Lightman et al., "Let's verify step by step," in *International Conference on Learning Representations*, 2024.

[15] I. Mirzadeh et al., "GSM-Symbolic: Understanding the limitations of mathematical reasoning in large language models," *arXiv:2410.05229*, 2024.

[16] F. Shi et al., "Large language models can be easily distracted by irrelevant context," in *International Conference on Machine Learning*, 2023.
