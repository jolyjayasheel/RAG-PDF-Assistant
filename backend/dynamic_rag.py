import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Disable Chroma telemetry before importing Chroma
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


# Windows console encoding
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")

        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")

    except Exception:
        pass


class DynamicRAG:
    """Dynamic RAG pipeline for uploaded documents."""

    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=os.getenv(
                "OLLAMA_BASE_URL",
                "http://host.docker.internal:11434"
            )
        )

        self.llm = ChatGroq(
            model=os.getenv(
                "GROQ_MODEL",
                "qwen/qwen3.8-27b"
            ),
            temperature=0
        )
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.current_document_name = None

    def create_vectorstore(self, documents, document_name):
        """Create Chroma vector store from document chunks."""
        try:
            # Delete previous vector store
            if self.vectorstore is not None:
                try:
                    self.vectorstore.delete_collection()
                except Exception:
                    pass
                self.vectorstore = None
            print(
                f"Creating vector store for '{document_name}'..."
            )

            for document in documents:

                document.metadata["source"] = document_name

            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            self.current_document_name = document_name

            print(
                f"[OK] Vector store created with "
                f"{len(documents)} chunks"
            )

            return True
        except Exception as e:
            raise Exception(
                f"Error creating vector store: {str(e)}"
            )

    def setup_retriever(self):
        """Setup similarity retriever with context compression."""

        if not self.vectorstore:
            raise Exception(
                "Vector store not initialized"
            )

        document_count = self.vectorstore._collection.count()

        k = min(5, document_count)

        print(
            f"[OK] Retriever configured with k={k}"
        )

        base_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k
            }
        )

        compressor = LLMChainExtractor.from_llm(
            self.llm
        )

        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        print(
            "[OK] Retriever + context compression setup complete"
        )

    def setup_qa_chain(self):
        """Setup RetrievalQA chain."""

        if not self.retriever:
            raise Exception(
                "Retriever not initialized"
            )

        prompt_template = """
You are a helpful assistant answering questions
about the uploaded document.

Use ONLY the provided document context to answer.

Document Context:
{context}

Question:
{question}

Instructions:
- Answer using only the provided context.
- Do not invent information.
- If the answer is not present in the context,
  say: "I don't have this information in the document."
- Give a clear and concise answer.
"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "context",
                "question"
            ]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt
            }
        )

        print(
            "[OK] QA chain setup complete"
        )

    # ======================================================
    # BUILD COMPLETE PIPELINE
    # ======================================================

    def build_pipeline(
        self,
        documents,
        document_name
    ):
        """Build complete RAG pipeline."""

        try:

            print(
                f"\nBuilding RAG pipeline for "
                f"'{document_name}'..."
            )

            # Step 1: Vector store
            self.create_vectorstore(
                documents,
                document_name
            )

            # Step 2: Retriever + compression
            self.setup_retriever()

            # Step 3: QA chain
            self.setup_qa_chain()

            print(
                f"[OK] RAG pipeline ready for "
                f"'{document_name}'!"
            )

            return True

        except Exception as e:

            raise Exception(
                f"Error building pipeline: {str(e)}"
            )

    def query(self, question):
        """Query the RAG pipeline."""

        if not self.qa_chain:
            raise Exception(
                "RAG pipeline not initialized"
            )

        try:

            print(
                f"\nQuerying: {question}"
            )

            # Run RAG pipeline
            result = self.qa_chain.invoke(
                {
                    "query": question
                }
            )

            sources = []

            for doc in result.get(
                "source_documents",
                []
            ):

                sources.append(
                    {
                        "document": doc.metadata.get(
                            "source",
                            self.current_document_name
                        ),

                        "page": doc.metadata.get(
                            "page"
                        ),

                        "content": doc.page_content
                    }
                )

            return {
                "answer": result["result"],

                "document": (
                    self.current_document_name
                ),

                "sources": sources
            }

        except Exception as e:

            import traceback

            traceback.print_exc()

            raise Exception(
                f"Error querying: {str(e)}"
            )

    def is_ready(self):
        """Check whether RAG pipeline is ready."""

        return self.qa_chain is not None