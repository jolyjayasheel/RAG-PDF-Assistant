from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import shutil
import os
from dotenv import load_dotenv
load_dotenv()

class DynamicRAG:
    """Dynamic RAG for uploaded documents"""
    
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
)
        
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.current_document_name = None
    
    def create_vectorstore(self, documents, document_name):
        """Create vector store from document chunks"""
        try:
            chroma_dir = "./chroma_db_dynamic"
            if os.path.exists(chroma_dir):
                shutil.rmtree(chroma_dir)
            print(f"Creating vector store for '{document_name}'...")

            self.vectorstore = Chroma.from_texts(
                texts=documents,
                embedding=self.embeddings,
                persist_directory=chroma_dir,
                metadatas=[{"source": document_name} for _ in documents]
            )
            self.current_document_name = document_name
            print(f"✓ Vector store created with {len(documents)} chunks")
            return True
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")
    
    def setup_retriever(self):
        """Setup retriever"""
        if not self.vectorstore:
            raise Exception("Vector store not initialized")
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        print("✓ Retriever setup complete")
    
    def setup_qa_chain(self):
        """Setup QA chain"""
        if not self.retriever:
            raise Exception("Retriever not initialized")  
        prompt_template = """You are a helpful assistant answering questions about the uploaded document.

Document Information:
{context}

Question: {question}

Answer based only on the provided document content. If you don't know the answer, say "I don't have this information in the document."
"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        print("✓ QA chain setup complete")
    
    def build_pipeline(self, documents, document_name):
        """Build complete RAG pipeline from chunks"""
        try:
            print(f"\nBuilding RAG pipeline for '{document_name}'...")
            
            self.create_vectorstore(documents, document_name)
            self.setup_retriever()
            self.setup_qa_chain()
            
            print(f"✓ RAG pipeline ready for '{document_name}'!")
            return True
        
        except Exception as e:
            raise Exception(f"Error building pipeline: {str(e)}")
    
    def query(self, question):
        """Query the RAG pipeline"""
        if not self.qa_chain:
            raise Exception("RAG pipeline not initialized")
        
        try:
            print(f"\nQuerying: {question}")
            result = self.qa_chain({"query": question})
            
            return {
                "answer": result["result"],
                "document": self.current_document_name,
                "source_chunks": [
                    doc.page_content for doc in result.get("source_documents", [])
                ]
            }
        except Exception as e:
            raise Exception(f"Error querying: {str(e)}")
    
    def is_ready(self):
        """Check if RAG pipeline is ready"""
        return self.qa_chain is not None