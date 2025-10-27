const cosine = (a, b) => {
  const dot = a.reduce((s, v, i) => s + v * b[i], 0);
  const magA = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
  const magB = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
  return dot / (magA * magB + 1e-10);
};

class InMemoryVectorStore {
  constructor() {
    this.items = []; // {id, page, text, embedding, summary}
  }

  add(id, page, text, embedding, summary = "") {
    this.items.push({ id, page, text, embedding, summary });
  }

  // return top-k by cosine similarity
  search(queryEmbedding, k = 3) {
    const scored = this.items.map((it) => ({
      ...it,
      score: cosine(it.embedding, queryEmbedding),
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  clear() {
    this.items = [];
  }
}

module.exports = new InMemoryVectorStore();
