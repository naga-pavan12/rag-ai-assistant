import os
import json
import hashlib
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Set base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
MAIN_DIR = os.path.dirname(BASE_DIR)  # project root

DATA_DIR = os.path.join(MAIN_DIR, "data", "jsonl_files")
VECTOR_DB_DIR = os.path.join(MAIN_DIR, "vectorstore_db")
EMBEDDED_IDS_FILE = os.path.join(MAIN_DIR, "embedded_ids.txt")

# Load the embedding model
print("🔄 Loading embedding model...")
model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# Setup Chroma DB
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = chroma_client.get_or_create_collection(name="product_data")

# Load already embedded IDs
if os.path.exists(EMBEDDED_IDS_FILE):
    with open(EMBEDDED_IDS_FILE, "r") as f:
        embedded_ids = set(f.read().splitlines())
else:
    embedded_ids = set()


def save_embedded_ids(new_ids):
    with open(EMBEDDED_IDS_FILE, "a") as f:
        for _id in new_ids:
            f.write(_id + "\n")


def generate_unique_id(filename, text):
    """Create a consistent unique ID based on filename + content."""
    hash_input = (filename + text).encode("utf-8")
    return hashlib.sha256(hash_input).hexdigest()


def embed_batch(texts, metadatas, ids):
    print(f"⏳ Embedding batch of size {len(texts)}...")
    embeddings = model.encode(texts, show_progress_bar=True)
    collection.add(documents=texts, metadatas=metadatas, ids=ids)
    return len(ids)


def embed_new_chunks():
    total_new = 0
    batch_size = 64  # Reduce batch size for memory safety

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(DATA_DIR, filename)
            print(f"\n📂 Processing file: {file_path}")

            with open(file_path, "r") as f:
                lines = f.readlines()

            new_ids, new_texts, metadatas = [], [], []

            for line in lines:
                try:
                    obj = json.loads(line)
                    text = obj.get("text", json.dumps(obj))
                    _id = generate_unique_id(filename, text)
                except Exception as e:
                    print(f"⚠️ Error parsing line: {e}")
                    continue

                if _id in embedded_ids:
                    continue

                new_ids.append(_id)
                new_texts.append(text)
                metadatas.append({"source": filename})

                if len(new_ids) >= batch_size:
                    total_new += embed_batch(new_texts, metadatas, new_ids)
                    save_embedded_ids(new_ids)
                    embedded_ids.update(new_ids)
                    new_ids, new_texts, metadatas = [], [], []

            # Final small batch
            if new_ids:
                total_new += embed_batch(new_texts, metadatas, new_ids)
                save_embedded_ids(new_ids)
                embedded_ids.update(new_ids)

    print(f"\n✅ Total new embeddings this run: {total_new}")


if __name__ == "__main__":
    print("🟢 Starting embedding process...")
    embed_new_chunks()
