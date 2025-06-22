import os
import json
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIG ---
jsonl_dir = "data/jsonl_files"
persist_directory = "vectorstore/db"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# --- EMBEDDING MODEL ---
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

# --- FLATTEN FUNCTION ---


def flatten_json(y, prefix=''):
    out = {}

    if isinstance(y, dict):
        for k, v in y.items():
            out.update(flatten_json(v, f"{prefix}{k}."))  # nested keys
    elif isinstance(y, list):
        for i, item in enumerate(y):
            out.update(flatten_json(item, f"{prefix}{i}."))  # list items
    else:
        out[prefix[:-1]] = str(y)  # final value
    return out

# --- JSON PARSER FOR ALL FILES ---


def auto_parse_jsonl(file_path):
    docs = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                json_obj = json.loads(line)
                flat_dict = flatten_json(json_obj)
                content = "\n".join(
                    [f"{k}: {v}" for k, v in flat_dict.items()])
                if content.strip():
                    docs.append(Document(page_content=content.strip(),
                                metadata={"source": file_path}))
            except Exception as e:
                print(f"⚠️ Skipping line in {file_path} due to error: {e}")
    return docs


# --- LOAD ALL JSONL FILES ---
all_docs = []
for file_name in os.listdir(jsonl_dir):
    if file_name.endswith(".jsonl"):
        file_path = os.path.join(jsonl_dir, file_name)
        all_docs.extend(auto_parse_jsonl(file_path))

print(f"📄 Total documents loaded: {len(all_docs)}")

# --- SPLIT INTO CHUNKS ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(all_docs)

print(f"✂️ Total chunks after splitting: {len(split_docs)}")

# --- BUILD VECTORSTORE ---
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    persist_directory=persist_directory
)
vectorstore.persist()

print(f"✅ Vectorstore created and saved to → {persist_directory}")
