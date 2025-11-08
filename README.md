# 🦙 Unsloth AI – Modern LLM Fine-Tuning & RL (5 Colabs)

This repo contains **5 Google Colab notebooks** using **Unsloth** to demonstrate:

* Full supervised fine-tuning
* LoRA / QLoRA parameter-efficient tuning
* Preference-based RL (DPO)
* GRPO-style reasoning RL
* Continued pretraining on a new language

All experiments run **entirely in Google Colab with GPU** and are explained in **one YouTube video**.

---

## 📺 Demo Video

Single walkthrough for all 5 Colabs:

* Demo: [Link](https://youtube.com)

---

## 📂 Notebooks

| # | Notebook file                                     | Topic                                |
| - | ------------------------------------------------- | ------------------------------------ |
| 1 | `colab1_full_finetune_smollm2.ipynb`              | Full fine-tuning (SmolLM2-135M)      |
| 2 | `colab2_lora_smollm2.ipynb`                       | LoRA / QLoRA fine-tuning             |
| 3 | `colab3_dpo_preference_rl.ipynb`                  | DPO preference RL                    |
| 4 | `colab4_grpo_reasoning_rl.ipynb`                  | GRPO-style reasoning RL              |
| 5 | `colab5_continued_pretraining_new_language.ipynb` | Continued pretraining (new language) |

Colab links (placeholders):

* Colab 1: [Link](https://colab.research.google.com/drive/1jYieqvd00MiSWXijzD5ybSxxqBYF-Ke7?usp=sharing)
* Colab 2: [Link](https://colab.research.google.com/drive/1uR6CC6vxwuZ7U8dlp43QhJkLV3dLrgwf?usp=sharing)
* Colab 3: [Link](https://colab.research.google.com/drive/1L5n3v1mhECB1nufJvwec5_hy-eFzrArB?usp=sharing)
* Colab 4: [Link](https://colab.research.google.com/drive/13mGU6kRFNKq-7DHtwTHWjvzCnpYJiTnz?usp=sharing)
* Colab 5: [Link](https://colab.research.google.com/drive/1tN3LaMJya3lNTrRALagAfIs-UHeMZLOP?usp=sharing)

---

## ⚙️ Environment (for all Colabs)

All notebooks assume:

* **Runtime:** Google Colab GPU (T4 or better)
* **Python:** Colab default (3.x)

Common install cell used in each notebook:

```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
from torch import __version__ as torch_version
from packaging.version import Version as V

xformers = "xformers==0.0.27" if V(torch_version) < V("2.4.0") else "xformers"
!pip install --no-deps {xformers} trl peft accelerate bitsandbytes datasets
```

---

## 📝 Colab 1 – Full Supervised Fine-Tuning (SmolLM2-135M)

**Link:** [Colab 1](https://colab.research.google.com/drive/1jYieqvd00MiSWXijzD5ybSxxqBYF-Ke7?usp=sharing)

**Goal**
Train a small model with **full supervised fine-tuning** (all weights trainable) on a tiny coding/chat dataset.

**Summary**

* **Model:** `unsloth/SmolLM2-135M-Instruct`
* **Technique:** `full_finetuning=True` via `FastLanguageModel.from_pretrained`, SFT with `trl.SFTTrainer`
* **Task:**

  * Simple coding and chat prompts (e.g., “write a Python function…”, “explain X in one paragraph”)
* **Data format:**

  * List of examples with `conversations = [{from: "human"/"gpt", value: "..."}]`
  * Converted to a `text` field using `tokenizer.apply_chat_template(...)`
* **Outputs:**

  * Finetuned checkpoint directory
  * Inference cell that answers coding/chat prompts in a helpful style

---

## 🧩 Colab 2 – LoRA / QLoRA Fine-Tuning (SmolLM2-135M)

**Link:** [Colab 2](https://colab.research.google.com/drive/1uR6CC6vxwuZ7U8dlp43QhJkLV3dLrgwf?usp=sharing)

**Goal**
Repeat Colab 1’s task using **LoRA with 4-bit loading** for parameter-efficient training.

**Summary**

* **Model:** `unsloth/SmolLM2-135M-Instruct` (4-bit)
* **Technique:**

  * Load with `load_in_4bit=True`
  * Apply LoRA via `FastLanguageModel.get_peft_model(...)`
  * Target modules: `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
* **Task & data:**

  * Same coding/chat dataset and chat template as Colab 1 for a fair comparison
* **Outputs:**

  * LoRA-only checkpoint directory
  * Inference helper that shows similar behavior to full FT with far fewer trainable parameters

---

## 🎯 Colab 3 – Preference RL with DPO (Prompt / Chosen / Rejected)

**Link:** [Colab 3](https://colab.research.google.com/drive/1L5n3v1mhECB1nufJvwec5_hy-eFzrArB?usp=sharing)

**Goal**
Demonstrate **Direct Preference Optimization (DPO)** using a tiny hand-crafted preference dataset.

**Summary**

* **Model:** `unsloth/SmolLM2-135M-Instruct` (4-bit + LoRA)
* **Technique:**

  * Unsloth’s `PatchDPOTrainer()` + `trl.DPOTrainer` + `trl.DPOConfig`
  * Training on tuples of (`prompt`, `chosen`, `rejected`)
* **Dataset:**

  * Small preference set such as:

    * Correct vs incorrect square function
    * Supportive vs dismissive response to a stressed student
* **Outputs:**

  * DPO-tuned checkpoint
  * Inference function (`dpo_chat`) that shows the model prefers the “chosen” style answers over the “rejected” ones

---

## 🧠 Colab 4 – GRPO-Style RL for Reasoning

**Link:** [Colab 4](https://colab.research.google.com/drive/13mGU6kRFNKq-7DHtwTHWjvzCnpYJiTnz?usp=sharing)

**Goal**
Implement a simple **GRPO-style RL loop** for **math reasoning** with a programmatic reward.

**Summary**

* **Model:** `unsloth/gemma-3-1b-it-unsloth-bnb-4bit` (or similar small reasoning model)
* **Task:**

  * Small arithmetic questions, for example:

    * “What is 12 + 7?”
    * “If you have 5 apples and buy 9 more, how many do you have now?”
* **Reward:**

  * Prompt format asks the model to end with `Final answer: <number>`
  * Reward = 1 if predicted number matches the ground truth, else 0
* **Loop:**

  * Generate K candidates per question
  * Compute rewards and normalize within the group
  * Compute GRPO-style loss using group-relative advantages and log-probs
* **Outputs:**

  * Reasoning-tuned checkpoint
  * Inference cell that prints questions and model answers, showing “Final answer: …” behavior

---

## 🌍 Colab 5 – Continued Pretraining in a New Language

**Link:** [Colab 5](https://colab.research.google.com/drive/1tN3LaMJya3lNTrRALagAfIs-UHeMZLOP?usp=sharing)

**Goal**
Show **continued pretraining (CPT)** of a model on text written in a **non-English language**.

**Summary**

* **Model:** `unsloth/SmolLM2-135M-Instruct`
* **Technique:**

  * Load with full weights (`load_in_4bit=False`, `full_finetuning=True`)
  * Train as a standard causal LM on a small custom corpus in a new language
* **Dataset:**

  * 10–20 sentences in the target language (e.g., Hindi)
  * Topics like daily life, exams, AI, etc.
* **Pipeline:**

  * Tokenize raw sentences with `max_seq_length`
  * Use `SFTTrainer` over a `text` field for next-token prediction
  * Generate sample outputs in the target language to verify behavior
* **Outputs:**

  * CPT checkpoint directory
  * Inference function that generates text in the new language

---

## ▶️ How to Use These Notebooks

1. Open any Colab link above.
2. Set **Runtime → Change runtime type → GPU**.
3. Run all cells from top to bottom:

   * Install dependencies
   * Load model and tokenizer
   * Prepare data
   * Train (SFT / DPO / GRPO / CPT)
   * Run inference for qualitative evaluation
4. Use the outputs and code walk-throughs in the **single demo video** linked at the top.

---

## ✅ Techniques Covered (Quick Recap)

* Full supervised fine-tuning of a small model
* LoRA / QLoRA parameter-efficient fine-tuning
* Preference-based RL with DPO (prompt / chosen / rejected)
* GRPO-style RL for reasoning tasks
* Continued pretraining on a new language corpus using Unsloth
