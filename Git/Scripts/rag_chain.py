from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from typing import List, Tuple
import chromadb

# ─── Configuration ─────────────────────────────────────────────
VDB_PATH = "vectorstore_database"
COLLECTIONS = ("product_embeddings", "prd_chunks")
TOP_K_PER_QUERY = 3
MAX_FINAL_RESULTS = 6


# ─── Query Type Detection ──────────────────────────────────────
def is_prd_prompt(user_query: str) -> bool:
    keywords = ["create a prd", "generate a prd", "write prd", "make a prd"]
    return any(kw in user_query.lower() for kw in keywords)


# ─── PRD Prompt Template ──────────────────────────────────────
def build_prd_prompt(user_query: str) -> str:
    return f"""
You are a senior product manager generating a PRD in the following format:

Title:
Brief:
Objective:
Problem Statement:
User Painpoints:
Assumptions and Dependencies:
Success Metrics:
User Stories & Acceptance Criteria:
Solution Overview:
Requirements:
Test Cases:
Edge Cases:
Impact Areas:
Notifications:
Permission Schema:
Data Migration:
Conclusion:

Please generate a detailed PRD based on this request:
{user_query}

Only return the PRD. Do not include any extra commentary.
""".strip()


# ─── Query Rewriting for Fusion ───────────────────────────────
def generate_fused_queries(query: str) -> List[str]:
    templates = [
        "{}",
        "Explain {}",
        "What are the details of {}?",
        "List all components of {}",
        "Give schema or structure for {}"
    ]
    return [template.format(query) for template in templates]


# ─── Run RAG-Fusion Search ─────────────────────────────────────
def run_rag_fusion(query: str) -> List[Document]:
    client = chromadb.PersistentClient(path=VDB_PATH)
    all_results = []

    fused_queries = generate_fused_queries(query)

    for cname in COLLECTIONS:
        col = client.get_collection(cname)
        seen = {}
        for fq in fused_queries:
            try:
                res = col.query(
                    query_texts=[fq],
                    n_results=TOP_K_PER_QUERY,
                    include=["documents", "distances"]
                )
                docs = res.get("documents", [[]])[0]
                scores = res.get("distances", [[]])[0]
                for doc, score in zip(docs, scores):
                    if doc not in seen:
                        seen[doc] = score
            except Exception as e:
                print(f"⚠️ RAG-Fusion failed on `{fq}` in `{cname}`: {e}")

        for doc, score in seen.items():
            all_results.append(Document(page_content=doc, metadata={
                               "source": cname, "score": score}))

    # Sort and return top results
    all_results.sort(key=lambda d: d.metadata.get("score", 999))
    return all_results[:MAX_FINAL_RESULTS]


# ─── Prompt Template for Normal RAG QA ─────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are Progress AI Assistant 1.0, a helpful product expert for a construction tech platform.

Use the below context to answer the user's question clearly and accurately.

If the answer isn't in the context, say:
"I couldn't find that information in the current project data."

--------------------
Context:
{context}

User Question:
{question}

Answer:
""".strip()
)


# ─── Main Unified Entry for Streamlit ──────────────────────────
def run_query_with_debug(user_query: str) -> Tuple[str, List[Document], List[str]]:
    llm = Ollama(model="mistral")

    if is_prd_prompt(user_query):
        prd_prompt = build_prd_prompt(user_query)
        chain = LLMChain(
            llm=llm, prompt=PromptTemplate.from_template("{query}"))
        answer = chain.run({"query": prd_prompt}).strip()
        return answer, [], ["prd_prompt_only"]

    # Else: RAG mode
    docs = run_rag_fusion(user_query)
    context_text = "\n---\n".join(d.page_content for d in docs)
    prompt_vars = {"context": context_text, "question": user_query}

    rag_chain = LLMChain(llm=llm, prompt=RAG_PROMPT)
    response = rag_chain.run(prompt_vars).strip()

    collections_used = list(
        set([d.metadata.get("source", "unknown") for d in docs]))
    return response, docs, collections_used
