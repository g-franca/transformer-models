from transformers import pipeline

def run_nlp_pipeline():
    print("--- 1. Análisis de Sentimiento (Sentiment Analysis) ---")
    # Inicializa el pipeline para clasificar texto
    classifier = pipeline("sentiment-analysis")
    
    texto_positivo = "I absolutely love learning about artificial intelligence!"
    texto_negativo = "Traffic was terrible today, and I am very frustrated."
    
    # Procesar los textos
    print(f"Texto: '{texto_positivo}' -> Resultado: {classifier(texto_positivo)}")
    print(f"Texto: '{texto_negativo}' -> Resultado: {classifier(texto_negativo)}\n")


    print("--- 2. Generación de Texto (Text Generation) ---")
    # Inicializa el pipeline para continuar o crear texto
    generator = pipeline("text-generation", model="gpt2")
    
    prompt = "In the future, robots will help humans to"
    # Genera una respuesta limitando el tamaño máximo
    resultado_gen = generator(prompt, max_length=30, num_return_sequences=1)
    
    print(f"Prompt: '{prompt}'")
    print(f"Resultado generado:\n{resultado_gen[0]['generated_text']}\n")


    print("--- 3. Reconocimiento de Entidades Nombradas (NER) ---")
    # Inicializa el pipeline para identificar nombres, lugares, organizaciones, etc.
    ner_tagger = pipeline("ner", aggregation_strategy="simple")
    
    texto_ner = "Elon Musk founded SpaceX, and its headquarters are located in Hawthorne, California."
    entidades = ner_tagger(texto_ner)
    
    print(f"Texto: '{texto_ner}'")
    print("Entidades detectadas:")
    for entidad in entidades:
        print(f"  - {entidad['word']}: {entidad['entity_group']} (Confianza: {entidad['score']:.2f})")

if __name__ == "__main__":
    run_nlp_pipeline()
