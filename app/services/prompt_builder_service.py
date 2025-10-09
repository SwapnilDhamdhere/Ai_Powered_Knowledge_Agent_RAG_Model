class PromptBuilder:
    def __init__(self, system_role: str = None):
        self.system_role = system_role or (
            "You are a helpful AI assistant. "
            "Always return clean plain text only. "
            "Do not use Markdown, tables, checklists, headings, or special formatting. "
            "If you do not know the answer based on the context, reply 'NO_ANSWER'."
        )

    def build_prompt(self, context: str, query: str) -> list[dict]:
        """
        Returns the prompt payload for Ollama or another LLM.
        """
        return [
            {"role": "system", "content": self.system_role},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]

    def build_with_intent(self, context: str, query: str, intent: str) -> list[dict]:
        """
        Adds dynamic intent-based instructions to the prompt.
        """
        return [
            {"role": "system", "content": self.system_role},
            {"role": "user", "content": f"Intent: {intent}\nContext:\n{context}\n\nQuestion:\n{query}"}
        ]