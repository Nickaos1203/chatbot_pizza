import os
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader

os.environ["CHROMA_ENABLE_TELEMETRY"] = "false"

# 1. Configuration
PDF_FILES = [
    'docs/Allergenes-FPO-Enseigne-MAJ-Janv-2024.pdf',
    'docs/Allergenes-Pizza-Rhuys.pdf',
    'docs/marco-fuso-recipe-booklet---final.pdf',
    'docs/Pizza-booklet-French.pdf',
    'docs/Pizza-maison.pdf',
    'docs/Recette-pizza-au-fromage.pdf',
    'docs/Tableau-des-allergenes.pdf'
]

COLLECTION_NAME = "rag_collection"

EMBEDDING_MODEL = "mxbai-embed-large"


# 2. Chargement et découpage des documents PDF
def load_and_chunk_pdf(file_path, chunk_size=1000, chunk_overlap=200):
    print(f"Chargement du fichier : {file_path}")
    reader = PdfReader(file_path)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    print(f"{file_path} contient {len(text)} caractères.")

    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk_text = text[i : i + chunk_size]
        if chunk_text.strip():
            chunks.append(chunk_text)
    print(f"{len(chunks)} chunks créés pour {file_path}.")
    return chunks

# 3. Initialisation de ChromaDB et embedding
print("Initialisation de ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")

print("Initialisation de la fonction d'embedding via Ollama...")
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=EMBEDDING_MODEL,
)

print(f"Création ou chargement de la collection : {COLLECTION_NAME}")
collection = client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=ollama_ef
)

# 4. Traitement de chaque PDF
all_chunks = []
all_ids = []
all_metadatas = []

for pdf_index, pdf_file in enumerate(PDF_FILES):
    chunks = load_and_chunk_pdf(pdf_file)
    chunk_ids = [f"{os.path.basename(pdf_file)}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": os.path.basename(pdf_file)} for _ in chunks]

    all_chunks.extend(chunks)
    all_ids.extend(chunk_ids)
    all_metadatas.extend(metadatas)

print("Stockage des chunks dans ChromaDB avec metadata...")
collection.add(
    documents=all_chunks,
    ids=all_ids,
    metadatas=all_metadatas
)

print("\n--- Base de données vectorielle créée avec succès ! ---")
print(f"Nombre de documents stockés : {collection.count()}")

