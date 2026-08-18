# Implementing Tokenization and Analyzing Attention

## Project Overview
This project fulfills the lab requirements by tokenizing sentences and evaluating internal **Self-Attention mechanisms** within a transformer model. Using Winograd Schema pairs, we analyze how changing a single descriptive word shifts pronoun coreference tracking between physical objects.

## Technical Requirements
- **Python 3.8+**
- **PyTorch**
- **Hugging Face Transformers Library**

## Architecture & Implementation Steps
1. **Tokenization:** Text is processed via the WordPiece algorithm (`distilbert-base-uncased`), outputting sub-word components along with system indicators (`[CLS]`, `[SEP]`).
2. **Attention Extraction:** Models run with `output_attentions=True` configurations, generating raw multi-head matrices from the Encoder framework.
3. **Coreference Profiling:** The multi-layered attention outputs are averaged over all 6 structural layers and 12 attention heads to discover semantic connection scores.

## Analytical Findings

### Sentence A: *"The trophy did not fit in the suitcase because it was too big."*
- **Observed Behavior:** The pronoun `"it"` assigns a significantly high weight score directly to the token `"trophy"`. 
- **Explanation:** The model's pre-trained deep semantic layer captures context-driven physical reasoning; things that are too large fail to fit inside target spaces.

### Sentence B: *"The trophy did not fit in the suitcase because it was too small."*
- **Observed Behavior:** Modifying the ending descriptor to `"small"` immediately relocates the core attention peak away from `"trophy"` and focuses heavily onto `"suitcase"`.
- **Explanation:** The model dynamically recalculates long-range constraints, determining that structural smallness belongs logically to the container entity.

## How to Execute the Script
1. Clone the repository and install requirements:
   ```bash
   pip install torch transformers numpy
   ```
2. Execute the primary Python pipeline file:
   ```bash
   python nlp_pipeline.py
   ```
