## Appendix

**A. Prompts used verbatim.** All conditions share the system message "You are a helpful assistant." and the problem text is placed before the instruction clause, unmodified. Only the final sentence differs.

C0: Answer the question. Output only the final number on a single line, with no other text.

C1: Solve the problem. Show your calculation steps in full.

C2: Solve the problem. Show your calculation steps in full. End with one line in the form "Final answer: <number>".

C3: Solve the problem. Show your calculation steps in full. Put the final number inside \boxed{}, for example \boxed{42}.

C4: Solve the problem. Show only the essential calculation steps, as briefly as possible.

C5: Solve the problem. Show only the essential calculation steps, as briefly as possible. End with one line in the form "Final answer: <number>".

C6: Solve the problem. Show only the essential calculation steps, as briefly as possible. Put the final number inside \boxed{}, for example \boxed{42}.

**B. Extractor rules.** Both extractors were frozen before the final grading pass and applied to the stored raw text; no model was re-run to change a score.

Strict. For C2 and C5 the extractor takes the last match of "final answer" (optionally followed by "is", a colon, a dash, or markdown emphasis) on a single line, and reads the first number in the remainder of that line. For C3 and C6 it takes the last complete \boxed{...} group and reads the first number inside. For C0, C1 and C4 there is no format to enforce, so the strict score is defined to equal the lenient score.

Lenient. Tries the three formats in order — last boxed group, last answer line, last number — and returns the first value found. A trailing number is accepted only if it begins after 80% of the response's characters; responses of 60 characters or fewer are exempt from this positional rule, since otherwise a bare answer such as "14" would be rejected.

Number parsing. A full fraction is evaluated first (so \frac-style content is not split), then a full decimal, then the first number anywhere in the captured span, after stripping thousands separators, currency symbols, trailing periods and percent signs. Comparison to the gold value uses a relative tolerance of 1e-4.

**C. Decoding and infrastructure.** Greedy decoding with a single sample, at most 512 new tokens, left padding, bfloat16 weights. The three models were served concurrently, one per NVIDIA A800-80GB GPU, using Hugging Face Transformers. Problem order was fixed across conditions by seed 7506; the 200 test problems were sampled without replacement. Mean completion time for the full 4,200-generation grid was under 15 minutes; grading and the paired analysis run on CPU in seconds.

**D. Reproducibility.** Per-item records contain the model, condition, problem id, gold answer, raw response, token count, truncation flag, and the output of both extractors. The generation script, the frozen extractor, and the analysis script are released alongside this report, so every number in the tables can be recomputed from the logs without GPU access.
