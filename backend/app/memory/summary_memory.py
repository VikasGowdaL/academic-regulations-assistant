
import os

from dotenv import load_dotenv
from langchain_cohere import ChatCohere

load_dotenv()


class SummaryMemory:

    def __init__(self):
        self.summary = ""
        self.messages = []

        self.llm = ChatCohere(
            model="command-a-03-2025",
            temperature=0,
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )

    def add_message(self, role, content):

        self.messages.append({
            "role": role,
            "content": content
        })

    def update_summary(self):

        if not self.messages:
            return self.summary

        conversation = "\n".join(
            f"{message['role'].capitalize()}: "
            f"{message['content']}"
            for message in self.messages
        )

        prompt = f"""
        Update the following conversation summary.

        Existing summary:
        {self.summary if self.summary else "No summary yet."}

        New conversation messages:
        {conversation}

        Instructions:
        1. Preserve important facts and context.
        2. Remove unnecessary details.
        3. Keep the summary concise.
        4. Do not invent information.

        Return only the updated summary.
        """

        response = self.llm.invoke(prompt)

        self.summary = response.content.strip()

        # Clear messages after summarization
        self.messages = []

        return self.summary

    def get_summary(self):
        return self.summary

    def clear(self):
        self.summary = ""
        self.messages = []


if __name__ == "__main__":

    memory = SummaryMemory()

    memory.add_message(
        "user",
        "What are the university attendance requirements?"
    )

    memory.add_message(
        "assistant",
        "The attendance rules are described in the academic regulations."
    )

    memory.add_message(
        "user",
        "What happens if I fail to meet the requirement?"
    )

    memory.add_message(
        "assistant",
        "We need to check the applicable university regulation."
    )

    summary = memory.update_summary()

    print("\n========== CONVERSATION SUMMARY ==========")
    print(summary)
