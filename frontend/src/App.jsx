
import { useState } from "react";
import axios from "axios";
import { Send, Bot, User, BookOpen } from "lucide-react";
import "./App.css";

function App() {
  const [sessionId] = useState(() => crypto.randomUUID());
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!question.trim() || loading) return;

    const userQuestion = question.trim();

    setMessages((prev) => [
      ...prev,
      { role: "user", content: userQuestion },
    ]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await axios.post(
  "http://127.0.0.1:8000/chat",
  {
    question: userQuestion,
    store_type: "faiss",
    session_id: sessionId
  }
);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.data.answer,
          sources: response.data.sources,
        },
      ]);
    } catch (error) {
      console.error("Chat API error:", error);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Unable to connect to the backend. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="brand-icon">
          <BookOpen size={25} />
        </div>

        <div>
          <h2>Academic Regulations Assistant</h2>
          <p>PES University regulations companion</p>
        </div>
      </header>

      <main className="chat-container">
        <div className="messages">
          {messages.length === 0 && (
            <div className="welcome">
              <Bot size={48} />

              <h2>How can I help you?</h2>

              <p>
                Ask me about attendance, academic calendars,
                curricula and examination policies.
              </p>
            </div>
          )}

          {messages.map((message, index) => (
            <div
              className={`message ${message.role}`}
              key={index}
            >
              <div className="message-icon">
                {message.role === "user" ? (
                  <User size={19} />
                ) : (
                  <Bot size={19} />
                )}
              </div>

              <div className="message-content">
                <p>{message.content}</p>

                {message.sources?.length > 0 && (
                  <div className="sources">
                    <strong>Sources</strong>

                    {message.sources.map((source, i) => (
                      <div key={i}>
                        {source.source} — Page {source.page}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="loading">
              <Bot size={19} />
              Assistant is thinking...
            </div>
          )}
        </div>

        <div className="input-area">
          <input
            type="text"
            placeholder="Ask about university regulations..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") sendMessage();
            }}
            disabled={loading}
          />

          <button
            onClick={sendMessage}
            disabled={loading || !question.trim()}
          >
            <Send size={19} />
          </button>
        </div>
      </main>
    </div>
  );
}

export default App;
