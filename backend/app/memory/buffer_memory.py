
class BufferMemory:

    def __init__(self, max_messages=10):
        self.max_messages = max_messages
        self.messages = []

    def add_message(self, role, content):

        self.messages.append({
            "role": role,
            "content": content
        })

        # Keep only the most recent messages
        self.messages = self.messages[-self.max_messages:]

    def get_history(self):

        if not self.messages:
            return "No previous conversation."

        history = []

        for message in self.messages:

            role = message["role"].capitalize()

            history.append(
                f"{role}: {message['content']}"
            )

        return "\n".join(history)

    def clear(self):
        self.messages = []


if __name__ == "__main__":

    memory = BufferMemory(max_messages=4)

    memory.add_message(
        "user",
        "What is the attendance requirement?"
    )

    memory.add_message(
        "assistant",
        "The attendance rules are described in the academic regulations."
    )

    memory.add_message(
        "user",
        "What happens if I fail to meet it?"
    )

    memory.add_message(
        "assistant",
        "Let me check the relevant attendance regulations."
    )

    print("========== CONVERSATION HISTORY ==========")
    print(memory.get_history())
