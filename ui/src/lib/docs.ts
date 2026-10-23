/** Human-readable names for the corpus documents (the ids are internal). */
export const DOC_LABELS: Record<string, string> = {
  lewis2020_rag: "Lewis et al., RAG",
  asai2023_selfrag: "Asai et al., Self-RAG",
  singhvi2023_dspy_assertions: "Singhvi et al., DSPy Assertions",
  pinecone_vector_db: "Pinecone: What is a vector database?",
  weaviate_docs: "Weaviate documentation",
  langchain_overview: "LangChain overview",
  zhu2023_pointcloud_survey: "Point-cloud augmentation survey (distractor)",
};

export const docLabel = (id: string): string => DOC_LABELS[id] ?? id.replace(/_/g, " ");
