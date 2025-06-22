# src/memory.py
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import Ollama
from langchain.docstore.document import Document


class ConversationMemory:
    def __init__(self):
        self.messages = []  # Full message history
        self.summaries = []  # Condensed summaries
        self.llm = Ollama(model="mistral")

    def add_user_message(self, msg):
        self.messages.append({"role": "user", "content": msg})

    def add_assistant_message(self, msg):
        self.messages.append({"role": "assistant", "content": msg})

    def summarize_if_needed(self):
        # Only summarize if > 6 messages are present and not already summarized
        if len(self.messages) >= 6 and len(self.summaries) == 0:
            text_blocks = [
                f"User: {m['content']}" if m['role'] == "user" else f"Assistant: {m['content']}"
                for m in self.messages[-6:]
            ]
            combined = "\n".join(text_blocks)
            doc = Document(page_content=combined)
            chain = load_summarize_chain(self.llm, chain_type="stuff")
            summary = chain.run([doc])
            self.summaries.append(summary)

    def get_memory_context(self):
        return "\n---\n".join(self.summaries)

    def reset(self):
        self.messages = []
        self.summaries = []
