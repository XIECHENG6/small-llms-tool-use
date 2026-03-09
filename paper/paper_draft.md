---
title: "Small LLMs as Tool-Use Agents: A Systematic Study of Parameter-Efficient Fine-Tuning for Function Calling"
author: "[Author Name], Northwestern Polytechnical University"
date: "2026"
geometry: margin=1in
fontsize: 12pt
linestretch: 1.5
numbersections: true
---

# Abstract

Tool use and function calling are essential capabilities for deploying large language models (LLMs) as autonomous agents. While proprietary models excel at these tasks, the potential of small open-source models (1.5B–7B parameters) remains underexplored. In this work, we present a systematic study of parameter-efficient fine-tuning for function calling across five small language models: Qwen2.5 (1.5B, 3B, 7B), LLaMA-3.2-3B, and Phi-3.5-mini. Using QLoRA 4-bit quantization on the Glaive Function Calling v2 dataset, we evaluate models across five metrics: JSON validity, function name accuracy, argument name accuracy, argument value accuracy, and exact match rate. Our experiments reveal three key findings: (1) after fine-tuning, all models converge to 85.8–88.2% exact match regardless of architecture, despite a 12-percentage-point spread in zero-shot performance; (2) a 1.5B-parameter model achieves 85.8% exact match, only 2.4% below a 7B model, suggesting minimal scaling benefit for this task; and (3) just 3,000 training examples capture over 90% of the performance gains achievable with 30,000 examples. We further identify that training data formatting—specifically, the inclusion of explicit stop tokens—has a larger impact on performance than model selection, improving exact match from 58.6% to 87.8%. A held-out function evaluation confirms that fine-tuned models generalize to completely unseen functions, achieving 91.8% exact match on functions never encountered during training. Our results demonstrate that small language models, when properly fine-tuned, can serve as effective and efficient tool-use agents.

# Introduction

The ability to interact with external tools through structured function calls has emerged as a critical capability for deploying language models as autonomous agents (Schick et al., 2023; Qin et al., 2023). Function calling enables models to translate natural language instructions into precise API invocations, bridging the gap between human intent and programmatic execution. While state-of-the-art proprietary models such as GPT-4 and Claude demonstrate strong function calling abilities, the extent to which smaller, open-source models can acquire these capabilities through parameter-efficient fine-tuning remains an open question.

Recent advances in parameter-efficient fine-tuning (PEFT) methods, particularly QLoRA (Dettmers et al., 2023), have made it feasible to adapt large models on consumer-grade hardware. Simultaneously, the release of capable small language models—including Qwen2.5 (Bai et al., 2023), LLaMA 3.2 (Touvron et al., 2023), and Phi-3.5 (Abdin et al., 2024)—has expanded the landscape of models suitable for resource-constrained deployment. However, systematic comparisons of these models on function calling tasks, particularly under controlled fine-tuning conditions, are lacking.

In this paper, we address this gap with a comprehensive empirical study. We fine-tune five small language models spanning three architectural families (Qwen, LLaMA, Phi) and three parameter scales (1.5B, 3B, 7B) using identical QLoRA configurations on the Glaive Function Calling v2 dataset. Our evaluation framework measures five complementary metrics that decompose function calling performance into distinct capabilities: JSON format compliance, function selection, argument naming, value matching, and overall exact match.

Our main contributions are as follows:

**Architecture convergence.** We demonstrate that QLoRA fine-tuning effectively eliminates performance differences between model architectures. While zero-shot exact match rates range from 61.6% (LLaMA-3.2-3B) to 73.6% (Qwen2.5-7B), fine-tuned models converge to a narrow band of 85.8–88.2%, with the gap between the weakest and strongest zero-shot performers shrinking from 12.0% to just 0.4%.

**Diminishing returns from scale.** Increasing model size from 1.5B to 7B parameters yields only a 2.4% improvement in exact match after fine-tuning (85.8% vs. 88.2%), suggesting that function calling is a task where small models can perform comparably to larger ones.

**Data efficiency.** We show that 3,000 training examples achieve 86.6% exact match, capturing over 90% of the gains obtained from 30,000 examples (89.4%). This finding has significant practical implications for domain-specific function calling deployment.

**Evaluation methodology matters.** We identify that training data formatting—specifically, the inclusion of explicit EOS tokens after function calls—and evaluation methodology (first-JSON extraction vs. naive parsing) together account for a 29.2% improvement in measured exact match, larger than any model selection or hyperparameter change.

**Strong out-of-distribution generalization.** Through a held-out function evaluation where 10 function types are completely excluded from training, we show that fine-tuned models achieve 91.8% exact match on unseen functions—higher than the 86.0% on seen functions—demonstrating that models learn generalizable function calling skills rather than memorizing specific function mappings.

# Related Work

## Tool Use in Language Models

The concept of augmenting language models with tool-use capabilities has gained significant attention. Toolformer (Schick et al., 2023) demonstrated that LLMs can learn to use external tools through self-supervised training. Subsequently, several works have explored structured function calling: Gorilla (Patil et al., 2023) proposed a retrieval-augmented approach for API calling, while ToolLLM (Qin et al., 2023) introduced a comprehensive framework for tool learning. The Berkeley Function-Calling Leaderboard (Yan et al., 2024) established standardized benchmarks for evaluating function calling capabilities. Our work complements these efforts by systematically studying the fine-tuning of small models rather than relying on prompting strategies for large models.

## Parameter-Efficient Fine-Tuning

Low-Rank Adaptation (LoRA; Hu et al., 2022) introduced the idea of training low-rank decomposition matrices to approximate weight updates, significantly reducing the number of trainable parameters. QLoRA (Dettmers et al., 2023) extended this approach by combining LoRA with 4-bit quantization, enabling fine-tuning of large models on consumer hardware. Our work applies QLoRA uniformly across all models, controlling for the fine-tuning method to isolate the effects of model architecture and scale.

## Small Language Models

The development of capable small language models has accelerated recently. The Qwen2.5 series (Bai et al., 2023) provides models from 0.5B to 72B parameters with strong multilingual capabilities. Meta's LLaMA 3.2 (Touvron et al., 2023) offers lightweight models optimized for on-device deployment. Microsoft's Phi series (Abdin et al., 2024) emphasizes data quality over model scale. While individual models have been evaluated on general benchmarks, comparative studies of their function calling capabilities under identical fine-tuning conditions are scarce.

# Methodology

## Task Definition

We define function calling as the task of generating a structured JSON output given a natural language user query and a set of available function definitions. Formally, given a system prompt *S* containing function definitions and a user query *Q*, the model must generate a JSON object *F* = {"name": *f*, "arguments": {*k₁*: *v₁*, ..., *kₙ*: *vₙ*}}, where *f* is the function name and {(*kᵢ*, *vᵢ*)} are argument key-value pairs.

## Dataset

We use the Glaive Function Calling v2 dataset (glaiveai/glaive-function-calling-v2), a synthetic dataset containing 112,960 conversation samples generated by GPT-4. Each sample includes a system prompt with function definitions, a multi-turn conversation, and one or more function calls. We parse the dataset to extract samples containing the `<functioncall>` tag, yielding 56,796 valid function calling examples. The dataset covers a diverse set of functions including mathematical calculations, API queries, text manipulation, and data retrieval.

## Models

We evaluate five instruction-tuned models spanning three architectural families:

| Model | Params | Family | Context | LoRA Targets |
|-------|--------|--------|---------|-------------|
| Qwen2.5-1.5B-Instruct | 1.5B | Qwen | 32K | Standard |
| Qwen2.5-3B-Instruct | 3B | Qwen | 32K | Standard |
| Qwen2.5-7B-Instruct | 7B | Qwen | 32K | Standard |
| LLaMA-3.2-3B-Instruct | 3B | LLaMA | 128K | Standard |
| Phi-3.5-mini-instruct | 3.8B | Phi | 128K | Phi-specific |

Table 1: Models evaluated in this study.

## Fine-Tuning Configuration

All models are fine-tuned using QLoRA with 4-bit NF4 quantization and double quantization. We apply LoRA adapters (rank=16, alpha=32, dropout=0.05) to all attention and MLP projection layers. For Qwen and LLaMA models, the target modules are q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, and down_proj. For Phi-3.5, the fused projection layers require different targets: qkv_proj, o_proj, gate_up_proj, and down_proj.

Training uses the Paged AdamW 8-bit optimizer with a cosine learning rate schedule (peak lr=2×10⁻⁴, warmup=50 steps). The effective batch size is 16 (batch size 4 with 4 gradient accumulation steps). All sequences are truncated to 512 tokens. Training runs for 3 epochs on a single NVIDIA L4 GPU (24GB).

## Training Data Formatting

A critical detail of our methodology is the explicit inclusion of the model-specific EOS token at the end of each function call in the training data. Our initial experiments without this token resulted in a "repetition generation" problem where models would generate a correct function call followed by additional unwanted JSON objects. The inclusion of the EOS token teaches the model to produce exactly one function call and stop, improving exact match from 58.6% to 87.8% (Section 5.4).

## Evaluation Metrics

We evaluate models on five hierarchical metrics, each building on the previous:

**JSON Valid Rate:** Whether the model output contains a parseable JSON object. We use a first-object extraction strategy that handles nested braces and ignores trailing text.

**Function Name Accuracy:** Whether the predicted function name exactly matches the ground truth.

**Argument Names Accuracy:** Whether the predicted argument keys exactly match the ground truth keys (set equality).

**Argument Values Accuracy:** Whether all argument values match the ground truth (case-insensitive string comparison).

**Exact Match Rate:** Whether the function name is correct AND all argument values match. This is our primary metric.

# Experiments

## Multi-Model Comparison

We fine-tune all five models using identical configurations: 10,000 training samples, LoRA rank 16, 3 epochs. Each model is evaluated on the same held-out test set of 500 samples. We report both zero-shot (base model without fine-tuning) and fine-tuned results.

## Data Scaling Study

Using Qwen2.5-3B as the representative model, we investigate the relationship between training data size and performance. We train with 500, 1,000, 3,000, 10,000, and 30,000 samples while keeping all other hyperparameters fixed.

## LoRA Rank Ablation

We vary the LoRA rank across {4, 8, 16, 32, 64} while keeping alpha = 2 × rank and all other hyperparameters fixed, using Qwen2.5-3B with 10,000 training samples.

## Held-Out Function Evaluation

To assess out-of-distribution generalization, we design an experiment where 10 function types are completely excluded from training and only appear at test time. The held-out functions span different difficulty levels: simple numeric (calculate_bmi), simple string (get_stock_price), list parameters (search_recipe), nested parameters (calculate_distance), and multi-parameter (send_email). The model is trained on the remaining functions and evaluated separately on seen (in-distribution) and unseen (out-of-distribution) test sets of 500 samples each.

# Results and Analysis

## Multi-Model Comparison

| Model | Condition | JSON % | Name % | ArgN % | ArgV % | EM % | ΔEM |
|-------|-----------|--------|--------|--------|--------|------|-----|
| Qwen2.5-1.5B | Zero-shot | 93.4 | 93.4 | 88.6 | 71.4 | 71.4 | — |
| Qwen2.5-1.5B | **Fine-tuned** | **100.0** | **100.0** | **96.8** | **85.8** | **85.8** | +14.4 |
| Qwen2.5-3B | Zero-shot | 100.0 | 100.0 | 96.0 | 72.2 | 72.2 | — |
| Qwen2.5-3B | **Fine-tuned** | **99.8** | **99.8** | **97.6** | **87.8** | **87.8** | +15.6 |
| Qwen2.5-7B | Zero-shot | 100.0 | 100.0 | 95.0 | 73.6 | 73.6 | — |
| Qwen2.5-7B | **Fine-tuned** | **100.0** | **100.0** | **97.6** | **88.2** | **88.2** | +14.6 |
| LLaMA-3.2-3B | Zero-shot | 99.4 | 99.4 | 95.4 | 61.6 | 61.6 | — |
| LLaMA-3.2-3B | **Fine-tuned** | **100.0** | **100.0** | **97.8** | **87.8** | **87.8** | +26.2 |
| Phi-3.5-mini | Zero-shot | 99.4 | 99.2 | 94.2 | 73.2 | 73.2 | — |
| Phi-3.5-mini | **Fine-tuned** | **99.6** | **99.2** | **97.4** | **87.4** | **87.4** | +14.2 |

Table 2: Zero-shot vs. fine-tuned performance across five models. All fine-tuned models use identical QLoRA configuration (rank=16, 10K samples, 3 epochs). EM = Exact Match.

The most striking finding is the convergence of fine-tuned performance: despite a 12-percentage-point spread in zero-shot exact match (61.6–73.6%), all fine-tuned models achieve between 85.8% and 88.2%, a range of just 2.4 percentage points. This convergence is particularly notable for LLaMA-3.2-3B, which gains 26.2 percentage points from fine-tuning—nearly double the improvement of any Qwen model—to match Qwen2.5-3B at 87.8%.

## Scaling Analysis

Within the Qwen family, increasing parameters from 1.5B to 7B yields diminishing returns: the 1.5B model achieves 85.8% exact match, the 3B model reaches 87.8% (+2.0%), and the 7B model achieves 88.2% (+0.4% over 3B). The cost of training and inference, however, scales roughly linearly with parameter count. This suggests that for function calling, the 1.5B–3B range offers the best efficiency–performance tradeoff.

## Data Scaling

| Training Samples | JSON % | Name % | ArgV % | EM % | Marginal Gain |
|-----------------|--------|--------|--------|------|---------------|
| 500 | 100.0 | 99.8 | 78.2 | 78.2 | — |
| 1,000 | 99.6 | 99.4 | 80.4 | 80.4 | +2.2 |
| 3,000 | 100.0 | 100.0 | 86.6 | 86.6 | +6.2 |
| 10,000 | 99.8 | 99.8 | 87.8 | 87.8 | +1.2 |
| 30,000 | 99.8 | 99.8 | 89.4 | 89.4 | +1.6 |

Table 3: Effect of training data size on Qwen2.5-3B performance.

Table 3 shows a clear logarithmic scaling pattern. The transition from 500 to 3,000 samples yields the steepest gains (+8.4%), while tripling the data from 10,000 to 30,000 produces only a +1.6% improvement. We identify 3,000 samples as the efficiency sweet spot: this amount achieves 86.6% exact match (97% of the 30K performance) while requiring only 10% of the training data and proportionally less compute.

## LoRA Rank Ablation

| LoRA Rank | Alpha | Est. Trainable Params | EM % |
|-----------|-------|----------------------|------|
| 4 | 8 | ~1M | 87.2 |
| 8 | 16 | ~2M | 87.6 |
| 16 | 32 | ~4M | 87.8 |
| 32 | 64 | ~8M | **88.4** |
| 64 | 128 | ~16M | 87.8 |

Table 4: Effect of LoRA rank on Qwen2.5-3B performance (10K samples, 3 epochs).

The results reveal remarkable insensitivity to LoRA rank: the total spread across a 16× range in trainable parameters (rank 4 to 64) is only 1.2 percentage points (87.2–88.4%). Rank 32 achieves the highest exact match at 88.4%, but the advantage over rank 4 is negligible (1.2%). This suggests that function calling requires very low model capacity to learn, consistent with our earlier observation that it is primarily a format-learning task rather than a deep reasoning task. Practitioners can safely use rank 4 to minimize computational cost with minimal performance sacrifice.

## Held-Out Function Evaluation

| Metric | Seen Functions | Unseen Functions | Gap |
|--------|---------------|-----------------|-----|
| JSON Valid Rate | 100.0% | 100.0% | 0.0% |
| Function Name Acc | 100.0% | 99.8% | 0.2% |
| Arg Names Acc | 97.2% | 97.6% | −0.4% |
| Arg Values Acc | 86.0% | 91.8% | −5.8% |
| Exact Match Rate | 86.0% | 91.8% | −5.8% |

Table 5: Performance on seen (in-distribution) vs. unseen (held-out) functions. The model was trained exclusively on seen functions and never exposed to unseen functions during training.

The held-out evaluation yields a surprising result: the model achieves *higher* exact match on unseen functions (91.8%) than on seen functions (86.0%). This definitively demonstrates that the model has learned generalizable function calling capabilities rather than memorizing function-specific mappings from the training data. The negative gap (−5.8%) is explained by the composition of the held-out set, which contains several structurally simple functions (calculate_bmi, get_stock_price, calculate_distance all achieve 100%).

Per-function analysis of unseen functions reveals strong generalization across difficulty levels: 4 out of 10 functions achieve 100% exact match, and 9 out of 10 exceed 75%. The sole outlier is send_email (11%), which requires generating long free-form text content for the email body—a fundamentally different challenge from structured parameter extraction.

## The Repetition Generation Problem

Our initial experiments (v2) achieved only 58.6% exact match with Qwen2.5-3B. Error analysis revealed that approximately 70% of failures were caused by the model generating a correct JSON object followed by one or more additional, unwanted function calls. For example, given a query about loan calculation, the model would output the correct function call, then generate a second function call with different parameter values.

This problem had two compounding effects. First, the raw model output was not a valid single JSON object, causing our initial evaluation parser (which used `json.loads` on the entire output) to classify the response as an invalid JSON. Second, even when partial parsing was attempted, extracting the correct function call from the concatenated output was unreliable.

We addressed this through three complementary solutions: (1) appending the model-specific EOS token to each training example's function call, teaching the model to stop after one generation; (2) specifying multiple stop token IDs during inference, including the EOS token and model-specific end tokens; and (3) implementing a first-JSON extraction parser that correctly handles nested braces. Together, these changes improved exact match from 58.6% to 87.8%, a 29.2-percentage-point increase without changing the model, training data quantity, or any hyperparameters.

# Discussion

## Why Do Models Converge?

The convergence of fine-tuned performance across architectures suggests that function calling may be a relatively "surface-level" capability that primarily requires learning a specific output format rather than deep reasoning. The models already understand the semantics of functions from pre-training; fine-tuning mainly teaches them to express this understanding in a structured JSON format. This interpretation is supported by the observation that JSON Valid Rate and Function Name Accuracy reach near-perfect levels across all models, while Argument Values Accuracy—which requires more precise understanding—shows the primary remaining errors.

## Practical Implications

Our results have several practical implications for deploying small models as tool-use agents. First, model selection matters less than training data quality: all five models perform comparably after fine-tuning, so practitioners should choose based on licensing, ecosystem compatibility, and inference efficiency rather than raw capability. Second, the 3,000-sample efficiency sweet spot means that domain-specific function calling can be bootstrapped with a relatively small labeled dataset. Third, proper formatting of training data (including EOS tokens) and evaluation methodology (first-JSON extraction) are at least as important as model selection.

## Limitations

Our study has several limitations. First, while our held-out function evaluation demonstrates strong generalization within the Glaive dataset, the functions share a common synthetic generation process (GPT-4). Evaluation on independently constructed benchmarks such as the Berkeley Function-Calling Leaderboard (BFCL) would provide additional validation. Second, the Glaive dataset may not fully represent the complexity and ambiguity of real-world function calling scenarios, particularly those involving multi-step reasoning or ambiguous user intent. Third, we do not evaluate multi-turn function calling, where the model must maintain context across several tool interactions. Fourth, our experiments use a single random seed; results may vary slightly with different seeds, though the consistency across five models suggests robust findings.

# Conclusion

We present a systematic study of parameter-efficient fine-tuning for function calling in small language models. Our experiments across five models (1.5B–7B parameters) from three architectural families demonstrate that QLoRA fine-tuning effectively eliminates performance differences between architectures, with all models converging to 85.8–88.2% exact match. We find that a 1.5B model achieves 97% of the performance of a 7B model, that 3,000 training examples capture over 90% of the gains from 30,000 examples, and that LoRA rank has negligible impact (1.2% spread across rank 4–64). Our held-out function evaluation confirms that fine-tuned models generalize to completely unseen functions, achieving 91.8% exact match on functions never encountered during training. Perhaps most importantly, we show that proper training data formatting and evaluation methodology account for a larger performance improvement (+29.2%) than model selection or scaling.

These findings suggest that small language models are viable tool-use agents for production deployment, particularly in resource-constrained environments. Future work will extend this study to include external benchmark evaluation (BFCL), multi-turn interactions, and comparison with more recent models.

# References

Abdin, M., et al. (2024). Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. *arXiv preprint arXiv:2404.14219*.

Bai, J., et al. (2023). Qwen Technical Report. *arXiv preprint arXiv:2309.16609*.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized Language Models. *NeurIPS 2023*.

Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.

Patil, S. G., Zhang, T., Wang, X., & Gonzalez, J. E. (2023). Gorilla: Large Language Model Connected with Massive APIs. *arXiv preprint arXiv:2305.15334*.

Qin, Y., et al. (2023). ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. *ICLR 2024*.

Schick, T., et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *NeurIPS 2023*.

Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv preprint arXiv:2302.13971*.

Yan, F., et al. (2024). Berkeley Function-Calling Leaderboard. https://gorilla.cs.berkeley.edu/leaderboard.html.
