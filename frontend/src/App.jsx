import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const uploadFile = async () => {
    if (!file) {
      setMessage("Please select a PDF first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setMessage("Uploading and processing document...");
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/api/upload",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      setMessage(
        `${data.document} uploaded successfully. ${data.chunks} chunks created.`
      );
    } catch (error) {
      setMessage(`❌ ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const askQuestion = async () => {
    if (!question.trim()) {
      setMessage("Please enter a question.");
      return;
    }

    setLoading(true);
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/api/query",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: question,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Question failed");
      }

      setAnswer(data.answer);
      setSources(data.sources || []);
      setMessage(`Answer generated from ${data.document}`);
    } catch (error) {
      setMessage(`❌ ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">

      <div className="container">

        <h1>📚 PDF RAG Assistant</h1>

        <p className="subtitle">
          Upload a document and ask questions about it.
        </p>

        {/* Upload Section */}

        <div className="card">

          <h2>📄 Upload Document</h2>

          <input
            type="file"
            accept=".pdf,.txt"
            onChange={(e) => setFile(e.target.files[0])}
          />

          <button
            onClick={uploadFile}
            disabled={loading}
          >
            {loading ? "Processing..." : "Upload Document"}
          </button>

        </div>

        {/* Question Section */}

        <div className="card">

          <h2>💬 Ask a Question</h2>

          <textarea
            placeholder="Ask something about your document..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />

          <button
            onClick={askQuestion}
            disabled={loading}
          >
            {loading ? "Thinking..." : "Ask Question"}
          </button>

        </div>

        {/* Status */}

        {message && (
          <div className="message">
            {message}
          </div>
        )}

        {/* Answer */}

        {answer && (
          <div className="card">

            <h2>🤖 Answer</h2>

            <p className="answer">
              {answer}
            </p>

          </div>
        )}

        {/* Sources */}

        {sources.length > 0 && (
          <div className="card">

            <h2>📖 Retrieved Sources</h2>

            {sources.map((source, index) => (
              <div className="source" key={index}>
                <strong>Source {index + 1}</strong>
                <p>{source}</p>
              </div>
            ))}

          </div>
        )}

      </div>

    </div>
  );
}

export default App;