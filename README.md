# Task-2-Genre-Classification
**Task 2** focuses on fine-tuning a language model to accurately classify short stories into 100 fine-grained genres based on a dataset of 1,000 samples (`FareedKhan/1k_stories_100_genre`).


## Inference & Execution Guide
1. Open in Google Colab: Upload the .ipynb notebook file to Google Colab, then upload the genre_classifier zip file.
2. Execute All Cells: Click Runtime ➔ Run all (Ctrl + F9).
3. Scroll to the last cell where the function `classify_story()` will classify the genre of the test set, or you can pass your story for classification.



## Performance
To address strict memory and compute constraints while maintaining low inference latency, I evaluated both lightweight Encoder models and quantized Decoder-based LLMs.
### ```distilbert-base-uncased```
  #### Training metrics:
  <img width="599" height="274" alt="DIstilledbert lora classifier train" src="https://github.com/user-attachments/assets/fc7cd943-d5ac-4c13-ac71-8f32334b73f2" /><br>
  Train set accuracy: **51.3%**
#### Test metrics:<br>
<img width="605" height="61" alt="DIstilledbert lora classifier test" src="https://github.com/user-attachments/assets/270f28a0-f57b-4910-9b98-ba53eb768812" /><br>
Test set accuracy: **44.9%**

### ```Qwen 2.5-3B``` (same model as Task 1 RAG with classification head)
#### Training metrics:
<img width="440" height="156" alt="Qwen classifictaion train" src="https://github.com/user-attachments/assets/8133e777-fa5d-4404-9f14-de8a493208aa" /><br>
Train set accuracy: **60%**

#### Test metrics:<br>
<img width="451" height="136" alt="Qwen classifictaion test" src="https://github.com/user-attachments/assets/29889641-e5bc-47b5-b913-ea90faf344f3" /><br>
Test set accuracy: **65%**


## Why DistilBERT over Qwen 2.5-3B for Production?
While `Qwen2.5-3B` achieved higher nominal top-1 accuracy (65.0%), its 3B parameters introduced massive computational overhead, making training sluggish and CPU inference non-viable for rapid prediction endpoints. In contrast, `distilbert-base-uncased` delivers hyper-fast CPU inference, fitting seamlessly into tight memory budgets without requiring dedicated GPU infrastructure for serving.



## Data Preprocessing & Chunking Strategy
### Stratified Dataset Splitting (80 / 10 / 10)
Because each genre contains only ~10 samples, a purely random split risks completely omitting certain classes from the validation or test sets.<br>
To prevent class distribution skew, I applied a two-stage stratified split using scikit-learn's train_test_split:
  * Stage 1 (Train vs. Temp): 80% assigned to training (train_idx), 20% held out to temporary evaluation set (temp_idx), stratified by labels.
  * Stage 2 (Val vs. Test): The 20% temporary set was split 50/50, yielding 10% Validation and 10% Test, also stratified by labels.
  * Outcome: Exactly balances the representation across all 100 genres (8 train, 1 validation, 1 test per genre), ensuring evaluation metrics evaluate every class uniformly.
### Overlapping Token Chunking
Because `distilbert-base-uncased` has a maximum sequence context of **512 tokens**, passing raw inputs would truncate crucial narrative context at the end of stories.

To overcome this, I implemented an overlapping sliding-window strategy:

  * Maximum Length: 512 tokens.

  * Stride: 128 tokens of overlap between adjacent chunks (`return_overflowing_tokens=True`).

#### Benefits:

  * Full Narrative Preservation: Ensures no story text is discarded.

  * In-Domain Data Expansion: Artificially increases the effective training set size without violating dataset modification restrictions.

  * Contextual Continuity: Overlapping tokens prevent semantic boundary loss across chunk transitions.

**Note:** on Long-Context Encoders: I experimented with `answerdotai/ModernBERT-base` to leverage its native 8,192-token sequence length without chunking. However, test accuracy remained nearly identical (~44.5%), confirming that dataset sparsity—not sequence truncation—was the main accuracy bottleneck.


## Training Methodology & Hyperparameters
### Why Use LoRA on a Small Encoder Model?
Although `DistilBERT` is inherently lightweight (~66M parameters), full fine-tuning on a 1,000-example dataset easily leads to catastrophic forgetting and weight distortion.

Applying Low-Rank Adaptation (LoRA) constrains parameter updates to small, low-rank matrices (r=8, α=16), acting as a strong structural regularizer that prevents overfitting while drastically reducing trainable parameters.

### Detailed Hyperparameter Configuration
#### DistilBERT Configuration
  * Base Model: distilbert/distilbert-base-uncased
  * LoRA Rank ($r$): 8 | Alpha ($\alpha$): 16 | Dropout: 0.1
  * Target Modules: q_lin, k_lin, v_lin, out_linL
  * earning Rate: 5e-5 (Linear decay)
  * Batch Size: 8 per device
  * Epochs: 10
  * Weight Decay: 0.01
  * Precision: FP16 mixed precision
  * Hardware & Runtime: Single NVIDIA T4 GPU (~18 minutes execution time)
  
#### Qwen2.5-3B QLoRA Configuration
  * Quantization: 4-bit NF4 (BitsAndBytesConfig) with double quantization and float16 compute.
  * LoRA Rank ($r$): 16 | Alpha ($\alpha$): 32 | Dropout: 0.1
  * Target Modules: All linear projections (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
  * Learning Rate: 3e-4
  * Effective Batch Size: 8 (`per_device_batch_size=2` $\times$ `gradient_accumulation_steps=4`)
  * Padding: Left-padding explicitly configured for decoder-based classification.
  * Hardware & Runtime: Single NVIDIA T4 GPU (~4 hours execution time)


## Qualitative Performance & Class Ambiguity Analysis
While a ~44% top-1 accuracy on DistilBERT appears modest on paper, evaluation of the predictions highlights significant genre taxonomy overlap across the 100 classes.<br>
<img width="544" height="209" alt="image" src="https://github.com/user-attachments/assets/64ffcdcf-a1ae-440a-91c0-77e43e29b9db" />

### Subtle Genre Boundaries & Human Benchmark
Many classes in the 100-genre taxonomy share near-identical semantic features:

  * Science Fiction vs. Hard Science Fiction

  * Paranormal vs. Supernatural

  * Dystopian vs. Post-Apocalyptic

Because these categories lack sharp semantic boundaries, a human annotator would similarly struggle to achieve high top-1 precision without strict definitions. This is reflected in **DistilBERT**’s **74.6%** Top-3 Accuracy and **Qwen**'s **91.0%** Top-5 Accuracy, confirming that the model consistently ranks the true genre within its top candidate predictions.



## System Trade-offs & Dual-Model Allocation Strategy
To build a modular, end-to-end NLP pipeline across Tasks 1 and 2 within limited compute environments, I balanced task-specific requirements against memory allocations:<br>
<img width="540" height="153" alt="Screenshot_133" src="https://github.com/user-attachments/assets/b842b317-0ceb-473a-a4b8-d673af8063c3" />
  * **Task 1** (Story Retrieval & Generation): Requires deep contextual reasoning and long-form text generation. I allocated the bulk of GPU VRAM to `Qwen2.5-3B-Instruct` and `ChromaDB`.

  * **Task 2** (Story Classification): Demands fast, repeatable, multi-class predictions. By selecting DistilBERT, Task 2 runs comfortably on CPU or minimal RAM budgets without contending for Task 1's GPU resources.

## Development Challenges & Solutions
|Challenge Encountered|Root Cause|Implemented Solution|
|--------|--------|----------------|
|Severe Overfitting|Extreme data sparsity (10 examples per genre across 100 classes).|Used LoRA parameter constraints (`r=8`) and overlapping sliding-window chunking to expand sample diversity.|
|Text Truncation|512-token limit cutting off narrative midstory.|Implemented `return_overflowing_tokens=True` with a 128-token stride during tokenization.|
|VRAM Exhaustion during Qwen Fine-Tuning|Sequence classification heads on large decoders require massive memory.|Applied 4-bit NF4 quantization (QLoRA) and gradient accumulation steps.|
|Decoder Classification Instability|Right-padding on decoder models distorts label logit alignment.|Set `tokenizer.padding_side = "left"` explicitly for decoder-based sequence classification.|


## What Would I Do Differently Next Time?
   * Hierarchical Multi-Label Classification: Instead of forcing a single choice among 100 flat classes, group overlapping genres into top-level parent categories (e.g., Speculative Fiction $\rightarrow$ Science Fiction $\rightarrow$ Hard Sci-Fi) using a hierarchical loss function.
   * Synthetic Data Augmentation: If dataset constraints were relaxed, leverage an LLM to generate synthetic variants for underrepresented genres to balance the class distribution.
