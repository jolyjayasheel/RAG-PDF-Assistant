import os
from pathlib import Path
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

class FileProcessor:
    """Process uploaded files (PDF, TXT) and chunk them"""
    
    def __init__(self):
        self.upload_dir = "./uploads"
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            
            return text
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")
    
    def extract_text_from_file(self, file_path):
        """Extract text from uploaded file (PDF or TXT)"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == ".pdf":
            return self.extract_text_from_pdf(file_path)
        elif file_extension == ".txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise Exception(f"Unsupported file type: {file_extension}")
    
    def chunk_text(self, text, chunk_size=500, chunk_overlap=100):
        """Split text into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        return chunks
    
    def process_file(self, file_path, chunk_size=500, chunk_overlap=100):
        """Process file and return chunks"""
        try:
            print(f"Extracting text from {file_path}...")
            text = self.extract_text_from_file(file_path)
            
            print(f"Chunking text (chunk_size={chunk_size})...")
            chunks = self.chunk_text(text, chunk_size, chunk_overlap)
            
            print(f"✓ Processed {len(chunks)} chunks from file")
            return chunks
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")
    
    def save_uploaded_file(self, file, filename):
        """Save uploaded file to disk"""
        file_path = os.path.join(self.upload_dir, filename)
        
        with open(file_path, 'wb') as f:
            f.write(file.file.read())
        
        return file_path