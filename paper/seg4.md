Figure 3. Capability tax relative to C1, in percentage points. Only the 7B model shows a detectable penalty, and it is largest when brevity and the boxed format are combined.

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
