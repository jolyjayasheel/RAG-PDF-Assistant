import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Disable Chroma telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


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

        print("[START] Loading HuggingFace embedding model...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32
            }
        )

        print("[OK] Embeddings model loaded!")

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

    # ======================================================
    # CREATE VECTOR STORE
    # ======================================================

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

            # Add document name to metadata
            for document in documents:
                document.metadata["source"] = document_name

            # Create Chroma vector store
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

    # ======================================================
    # SETUP RETRIEVER
    # ======================================================

    def setup_retriever(self):
        """Setup similarity-based retriever."""

        if not self.vectorstore:
            raise Exception(
                "Vector store not initialized"
            )

        document_count = self.vectorstore._collection.count()

        k = min(5, document_count)

        print(
            f"[OK] Retriever configured with k={k}"
        )

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k
            }
        )

        print(
            "[OK] Similarity retriever setup complete"
        )

    # ======================================================
    # SETUP QA CHAIN
    # ======================================================

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
  say exactly:
  "I don't have this information in the document."
- Give a clear and concise answer.
- When possible, preserve important details from the document.
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
        """Build the complete RAG pipeline."""

        try:

            print(
                f"\nBuilding RAG pipeline for "
                f"'{document_name}'..."
            )

            # Step 1: Create embeddings + Chroma
            self.create_vectorstore(
                documents,
                document_name
            )

            # Step 2: Configure retriever
            self.setup_retriever()

            # Step 3: Configure QA chain
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

    # ======================================================
    # QUERY
    # ======================================================

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

    # ======================================================
    # STATUS
    # ======================================================

    def is_ready(self):
        """Check whether RAG pipeline is ready."""

        return self.qa_chain is not None