import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { Send, Loader, Bot, User, Sparkles } from "lucide-react";
import "./ChatInterface.css";

interface Message {
  id: string;
  type: "user" | "bot";
  content: string;
  timestamp: Date;
  analogy?: string;
  explanation?: string;
  isError?: boolean;
}

interface AnalogyResponse {
  analogy: string;
  explanation: string;
  original_question: string;
  success: boolean;
  error_message?: string;
}

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const API_BASE_URL = "http://localhost:8000";

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);

    try {
      const response = await axios.post<AnalogyResponse>(
        `${API_BASE_URL}/generate-analogy`,
        {
          question: userMessage.content,
          difficulty_level: "medium",
        }
      );

      const data = response.data;

      if (data.success) {
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: "bot",
          content: `Here's an analogy for "${data.original_question}":`,
          analogy: data.analogy,
          explanation: data.explanation,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, botMessage]);
      } else {
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: "bot",
          content:
            "Sorry, I encountered an error while generating your analogy. Please try again!",
          timestamp: new Date(),
          isError: true,
        };
        setMessages((prev) => [...prev, errorMessage]);
      }
    } catch (error) {
      console.error("Error:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content:
          "Sorry, I couldn't connect to the server. Make sure the backend is running!",
        timestamp: new Date(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const renderMessage = (message: Message) => {
    return (
      <div key={message.id} className={`message ${message.type}`}>
        <div className="message-avatar">
          {message.type === "user" ? (
            <User size={16} />
          ) : (
            <div className="bot-avatar">
              <Sparkles size={16} />
            </div>
          )}
        </div>
        <div className="message-content">
          <div className={`message-bubble ${message.isError ? "error" : ""}`}>
            <p className="message-text">{message.content}</p>
            {message.analogy && (
              <div className="analogy-section">
                <div className="analogy">
                  <p>{message.analogy}</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const welcomePrompts = [
    "How does machine learning work?",
    "What is quantum physics?",
    "Explain blockchain technology",
    "How does the human brain process memories?",
  ];

  return (
    <div className="chat-container">
      {/* Header */}
      <div className="chat-header">
        <div className="header-content">
          <div className="header-logo">
            <Sparkles size={20} />
          </div>
          <h1 className="header-title">AnalogyGPT</h1>
        </div>
      </div>

      {messages.length === 0 && !isLoading && (
        <div className="welcome-section">
          <div className="welcome-header">
            <div className="logo">
              <Sparkles size={32} />
            </div>
            <h1>Welcome to AnalogyGPT</h1>
            <p>Turn complex ideas into simple, clever analogies</p>
          </div>

          <div className="example-prompts">
            <h3>Try asking about:</h3>
            <div className="prompt-grid">
              {welcomePrompts.map((prompt, index) => (
                <button
                  key={index}
                  className="prompt-button"
                  onClick={() => setInputValue(prompt)}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {messages.length > 0 && (
        <div className="messages-container">
          {messages.map(renderMessage)}
          {isLoading && (
            <div className="message bot">
              <div className="message-avatar">
                <div className="bot-avatar">
                  <Sparkles size={16} />
                </div>
              </div>
              <div className="message-content">
                <div className="message-bubble loading">
                  <Loader className="loading-spinner" size={16} />
                  <span>Creating your analogy...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      )}

      <div className="input-section">
        <form className="input-form" onSubmit={handleSubmit}>
          <div className="input-container">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me to explain anything with an analogy..."
              className="message-input"
              disabled={isLoading}
              rows={1}
            />
            <button
              type="submit"
              className="send-button"
              disabled={!inputValue.trim() || isLoading}
            >
              <Send size={18} />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;
