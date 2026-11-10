
from app.memory.buffer_memory import BufferMemory
from app.memory.summary_memory import SummaryMemory


class ConversationMemory:

    def __init__(self, max_messages=6):

        self.max_messages = max_messages

        self.buffer = BufferMemory(
            max_messages=max_messages
        )

        self.summary_memory = SummaryMemory()

        self.pending_messages = []

    def add_message(self, role, content):

        message = {
            "role": role,
            "content": content
        }

        # Store the message in recent conversation history
        self.buffer.add_message(role, content)

        self.pending_messages.append(message)

        # Summarize older messages when the limit is exceeded
        if len(self.pending_messages) > self.max_messages:

            older_messages = self.pending_messages[
                :-self.max_messages
            ]

            self.pending_messages = self.pending_messages[
                -self.max_messages:
            ]

            for old_message in older_messages:
                self.summary_memory.add_message(
                    old_message["role"],
                    old_message["content"]
                )

            self.summary_memory.update_summary()

    def get_context(self):

        summary = self.summary_memory.get_summary()

        recent_history = self.buffer.get_history()

        return (
            "PREVIOUS CONVERSATION SUMMARY:\n"
            + (summary if summary else "No previous summary.")
            + "\n\nRECENT CONVERSATION:\n"
            + recent_history
        )

    def clear(self):

        self.buffer.clear()
        self.summary_memory.clear()
        self.pending_messages = []


if __name__ == "__main__":

    memory = ConversationMemory(max_messages=6)

    memory.add_message(
        "user",
        "What are the attendance requirements?"
    )

    memory.add_message(
        "assistant",
        "Let us check the academic regulations."
    )

    memory.add_message(
        "user",
        "What happens if I fail to meet them?"
    )

    memory.add_message(
        "assistant",
        "We need to examine the applicable attendance policy."
    )

    memory.add_message(
        "user",
        "Can I take the examination?"
    )

    memory.add_message(
        "assistant",
        "That depends on the examination eligibility rules."
    )

    # This seventh message triggers summarization
    memory.add_message(
        "user",
        "Where can I find those rules?"
    )

    print("\n========== COMBINED MEMORY ==========")
    print(memory.get_context())
