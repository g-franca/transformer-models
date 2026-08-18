import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

def analyze_attention_mechanism(sentence, target_pronoun="it"):
    """
    Tokenizes a sentence and extracts attention weights to observe 
    how a transformer resolves pronoun coreferences.
    """
    # 1. Load pretrained encoder model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    
    # 2. Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # 3. Forward pass to collect model layer activations and attention matrices
    with torch.no_grad():
        outputs = model(**inputs)
    
    # outputs.attentions contains a tuple of tensors for each layer
    # Shape per layer tensor: (batch_size, num_heads, sequence_length, sequence_length)
    attention_layers = outputs.attentions
    
    # 4. Locate the target pronoun index
    try:
        pronoun_idx = tokens.index(target_pronoun)
    except ValueError:
        print(f"Target token '{target_pronoun}' not found in the tokenized sequence.")
        return None
        
    print(f"\n==================================================")
    print(f"Analyzing Sentence: '{sentence}'")
    print(f"Generated Tokens: {tokens}")
    print(f"Pronoun '{target_pronoun}' found at index: {pronoun_idx}")
    print(f"==================================================")
    
    # 5. Extract global attention distribution for the pronoun
    # Stack layers to shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
    stacked_attention = torch.stack(attention_layers)
    
    # Average across all layers (dim 0) and all heads (dim 2)
    mean_attention_matrix = stacked_attention.mean(dim=0).squeeze(0).mean(dim=0)
    
    # Isolate attention vectors specifically originating from our pronoun
    pronoun_attention_vector = mean_attention_matrix[pronoun_idx].numpy()
    
    # 6. Print attention scores targeting key sequence tokens
    print("Mean Self-Attention scores emitted from the pronoun:")
    for idx, token in enumerate(tokens):
        score = pronoun_attention_vector[idx]
        print(f"  '{target_pronoun}' ---> '{token}': {score:.4f}")
        
    return tokens, pronoun_attention_vector

if __name__ == "__main__":
    # Winograd Schema sentences to test context-driven pronoun resolution
    sentence_a = "The trophy did not fit in the suitcase because it was too big."
    sentence_b = "The trophy did not fit in the suitcase because it was too small."
    
    print("--- RUNNING CRITERIA EVALUATION ---")
    analyze_attention_mechanism(sentence_a)
    analyze_attention_mechanism(sentence_b)
