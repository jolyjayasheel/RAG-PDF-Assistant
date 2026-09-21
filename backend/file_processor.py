import os
from pathlib import Path
import PyPDF2

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FileProcessor:
    """Process uploaded files and create chunks with metadata."""

    def __init__(self):
        self.upload_dir = "./uploads"
        os.makedirs(self.upload_dir, exist_ok=True)

        self.chunk_size = 1000
        self.chunk_overlap = 150

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def extract_text_from_pdf(self, file_path):
        """Extract PDF text while preserving page numbers."""

        try:
            documents = []

            with open(file_path, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)

                for page_number, page in enumerate(pdf_reader.pages, start=1):
                    page_text = page.extract_text()

                    if page_text and page_text.strip():
                        documents.append(
                            Document(
                                page_content=page_text,
                                metadata={
                                    "source": Path(file_path).name,
                                    "page": page_number
                                }
                            )
                        )

            return documents

        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")

    def extract_text_from_file(self, file_path):
        file_extension = Path(file_path).suffix.lower()

        if file_extension == ".pdf":
            return self.extract_text_from_pdf(file_path)

        elif file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": Path(file_path).name,
                        "page": None
                    }
                )
            ]

        else:
            raise Exception(
                f"Unsupported file type: {file_extension}"
            )

    def chunk_documents(self, documents):
        """Split documents while preserving their metadata."""

        chunks = []

        for document in documents:
            page_chunks = self.text_splitter.split_documents([document])
            chunks.extend(page_chunks)

        return chunks

    def process_file(self, file_path):
        try:
            print(f"📄 Extracting text from: {file_path}")

            documents = self.extract_text_from_file(file_path)

            if not documents:
                raise Exception("No readable text found in the file.")

            print(
                f"✂️ Chunking text "
                f"(chunk_size={self.chunk_size}, "
                f"overlap={self.chunk_overlap})..."
            )

            chunks = self.chunk_documents(documents)

            total_characters = sum(
                len(doc.page_content) for doc in documents
            )

            print(f"✓ Extracted characters: {total_characters:,}")
            print(f"✓ Created chunks: {len(chunks):,}")

            return chunks

        except Exception as e:
            raise Exception(
                f"Error processing file: {str(e)}"
            )

    def save_uploaded_file(self, file, filename):
        file_path = os.path.join(self.upload_dir, filename)

        with open(file_path, "wb") as f:
            f.write(file.file.read())

        return file_path