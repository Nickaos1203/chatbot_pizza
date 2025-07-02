import os
import json
import requests
import chromadb
from chromadb.utils import embedding_functions
import gradio as gr

# Configuration
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2"
CHROMA_PATH = "./chroma_db"
OLLAMA_URL = "http://localhost:11434/api/chat"

# Initialisation
os.environ["CHROMA_ENABLE_TELEMETRY"] = "false"

print("Connexion à ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_PATH)

print("Chargement du modèle d'embedding via Ollama...")
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=EMBEDDING_MODEL,
)

print(f"Chargement de la collection : {COLLECTION_NAME}")
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ollama_ef
)

# Fonction RAG
def rag_chatbot_llm(query, top_k=3):
    if not query.strip():
        return "Veuillez entrer une question."

    print(f"🔍 Recherche des {top_k} chunks les plus pertinents...")
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    documents = results.get("documents", [[]])[0]

    if not documents:
        return "Aucun résultat trouvé."

    # Construction du contexte à partir des chunks récupérés
    context = "\n".join(documents)

    # Construction du prompt pour le LLM
    prompt = (
        "En te basant uniquement sur le contexte suivant, réponds de manière claire et complète à la question.\n\n"
        f"--- CONTEXTE ---\n{context}\n\n"
        f"--- QUESTION ---\n{query}"
    )

    # Appel au modèle LLM via Ollama
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        })

        if response.status_code == 200:
            return json.loads(response.text)["message"]["content"]
        else:
            return f"Erreur LLM : {response.text}"

    except Exception as e:
        return f"Erreur lors de l'appel au LLM : {str(e)}"


# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Chatbot RAG ")

    with gr.Row():
        question = gr.Textbox(label="Votre question", placeholder="Ex: Quels sont les ingrédients de la pizza hawaïenne ?")
        submit = gr.Button("Répondre")

    response_box = gr.Markdown(label="Réponse")

    submit.click(fn=rag_chatbot_llm, inputs=question, outputs=response_box)


# Lancement
if __name__ == "__main__":
    demo.launch()
