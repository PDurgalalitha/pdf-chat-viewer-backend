const express = require("express");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const pdfParse = require("pdf-parse");
const dotenv = require("dotenv");
const Groq = require("groq-sdk");
const { pipeline } = require("@xenova/transformers");
dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());
app.use("/uploads", express.static(path.join(__dirname, "uploads")));

const upload = multer({ storage: multer.memoryStorage() });
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// Temporary in-memory vector store
let pdfChunks = [];
let embedder = null;

// Load embedding model once at startup
(async () => {
  embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  console.log(" Local embedding model loaded successfully");
})();

// Split text into chunks (~1000 chars)
function chunkText(text, size = 1000) {
  const chunks = [];
  for (let i = 0; i < text.length; i += size) {
    chunks.push(text.slice(i, i + size));
  }
  return chunks;
}

// Compute cosine similarity
function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dot / (normA * normB);
}

app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!embedder) {
      return res.status(503).json({
        error: "Embedding model not ready yet. Try again in a few seconds.",
      });
    }

    // Save uploaded file temporarily for preview
    const uploadPath = path.join(__dirname, "uploads");
    if (!fs.existsSync(uploadPath)) fs.mkdirSync(uploadPath);
    const filePath = path.join(uploadPath, req.file.originalname);
    fs.writeFileSync(filePath, req.file.buffer);

    const pdfData = await pdfParse(req.file.buffer);
    const pages = pdfData.text
      .split("\f")
      .map((p) => p.trim())
      .filter(Boolean);

    // Split each page into chunks and tag with page number
    const chunks = [];
    pages.forEach((pageText, pageIndex) => {
      const subChunks = chunkText(pageText, 1000);
      subChunks.forEach((chunk) => {
        chunks.push({ text: chunk, page: pageIndex + 1 });
      });
    });

    pdfChunks = [];

    for (const chunkObj of chunks) {
      const output = await embedder(chunkObj.text, {
        pooling: "mean",
        normalize: true,
      });
      pdfChunks.push({
        text: chunkObj.text,
        vector: Array.from(output.data),
        page: chunkObj.page,
      });
    }

    res.json({
      message: "PDF uploaded and vectorized successfully.",
      totalChunks: pdfChunks.length,
      pdfUrl: `${BASE_URL}/uploads/${req.file.originalname}`,
    });
  } catch (error) {
    console.error("Error processing PDF:", error.message);
    res.status(500).json({ error: "Upload failed", details: error.message });
  }
});

// Route: Query using semantic similarity + Groq
app.post("/chat", async (req, res) => {
  try {
    const { question } = req.body;
    if (!pdfChunks.length) {
      return res.status(400).json({ error: "No PDF uploaded yet" });
    }

    if (!embedder) {
      return res.status(503).json({ error: "Embedding model not ready yet" });
    }

    // Create local embedding for the question
    const qEmbedding = await embedder(question, {
      pooling: "mean",
      normalize: true,
    });
    const questionEmbedding = Array.from(qEmbedding.data);

    // Rank chunks by similarity
    const ranked = pdfChunks
      .map((chunk) => ({
        ...chunk,
        score: cosineSimilarity(questionEmbedding, chunk.vector),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);

    const context = ranked.map((r) => r.text).join("\n---\n");

    // Ask Groq (free Llama3 model)
    const completion = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [
        {
          role: "system",
          content:
            "You are a helpful assistant answering questions about a document.",
        },
        {
          role: "user",
          content: `Use the following context to answer the question:\n${context}\n\nQuestion: ${question}`,
        },
      ],
      max_tokens: 300,
    });

    res.json({
      answer: completion.choices[0].message.content,
      citations: ranked.map((r) => ({
        preview: r.text.slice(0, 80) + "...",
        page: r.page,
      })),
    });
  } catch (error) {
    res.status(500).json({ error: "Query failed", details: error.message });
  }
});

const PORT = process.env.PORT || 4000;
const BASE_URL = process.env.BASE_URL;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
