from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, Any
import time
from pathlib import Path
from app.core.config import settings
from app.services.parsers import DocumentParser
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class DocumentService:
    """Service for processing and indexing documents"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.DOCUMENT_CHUNK_SIZE,
            chunk_overlap=settings.DOCUMENT_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " "]
        )
        self.parser = DocumentParser()
        self.embedding_service = EmbeddingService()
        self.pinecone_service = PineconeService()
    
    def process_document(self, file_path: str, filename: str, progress_callback=None) -> Dict[str, Any]:
        """
        Process uploaded document and store in vector database
        Optimized for speed and accuracy
        """
        process_start = time.time()
        
        try:
            step_start = time.time()
            if progress_callback:
                progress_callback("parsing", "Parsing document content...", 10)
            file_ext = Path(filename).suffix.lower()
            
            parse_start = time.time()
            if file_ext in ['.docx', '.doc']:
                content = self.parser.parse_docx(file_path, progress_callback)
            elif file_ext == '.pdf':
                content = self.parser.parse_pdf(file_path, progress_callback)
            elif file_ext == '.txt':
                content = self.parser.parse_txt(file_path, progress_callback)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            parse_duration = time.time() - parse_start
            
            if not content:
                raise ValueError("No content extracted from document")
            
            content_length = len(content) if isinstance(content, str) else sum(len(chunk.page_content) for chunk in content) if hasattr(content, '__iter__') else 0
            step_duration = time.time() - step_start
            if progress_callback:
                progress_callback("parsing", f"Parsed {content_length:,} characters", 20)
            
            # STEP 3b: Chunk the document
            step_start = time.time()
            if progress_callback:
                progress_callback("chunking", "Chunking document into smaller pieces...", 30)
            
            chunk_start = time.time()
            if isinstance(content, str):
                chunks = self.text_splitter.split_text(content)
            else:
                chunks = [chunk.page_content for chunk in self.text_splitter.split_documents(content)]
            chunk_duration = time.time() - chunk_start
            
            step_duration = time.time() - step_start
            total_chars = sum(len(chunk) for chunk in chunks)
            avg_chunk_size = total_chars / len(chunks) if chunks else 0
            if progress_callback:
                progress_callback("chunking", f"Created {len(chunks):,} chunks", 40)
            
            # STEP 3c: Prepare documents for embedding
            step_start = time.time()
            if progress_callback:
                progress_callback("preparing", "Preparing documents for embedding...", 45)
            
            prep_start = time.time()
            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "metadata": {
                        "filename": filename,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                })
            prep_duration = time.time() - prep_start
            
            step_duration = time.time() - step_start
            if progress_callback:
                progress_callback("preparing", f"Prepared {len(documents):,} documents", 50)
            
            # STEP 3d: Get Pinecone index
            step_start = time.time()
            self.pinecone_service.get_index()
            step_duration = time.time() - step_start
            if progress_callback:
                progress_callback("preparing", "Connected to Pinecone", 55)
            
            # STEP 3e: Generate embeddings
            step_start = time.time()
            if progress_callback:
                progress_callback("embedding", f"Generating embeddings for {len(documents):,} chunks...", 60)
            
            texts = [doc["text"] for doc in documents]
            all_embeddings = self.embedding_service.generate_embeddings(texts)
            
            step_duration = time.time() - step_start
            if progress_callback:
                progress_callback("embedding", f"Generated {len(all_embeddings):,} embeddings", 75)
            
            # STEP 3f: Prepare vectors
            step_start = time.time()
            vectors = self.pinecone_service.prepare_vectors(documents, all_embeddings, filename)
            step_duration = time.time() - step_start
            
            # STEP 3g: Upsert to Pinecone
            step_start = time.time()
            if progress_callback:
                progress_callback("indexing", f"Indexing {len(vectors):,} vectors...", 80)
            
            upserted_count = self.pinecone_service.upsert_vectors(vectors)
            
            step_duration = time.time() - step_start
            if progress_callback:
                progress_callback("indexing", f"Indexed {upserted_count:,} vectors", 100)
            
            total_duration = time.time() - process_start
            
            return {
                "success": True,
                "filename": filename,
                "chunks_processed": len(documents),
                "total_chars": total_chars
            }
            
        except Exception as e:
            total_duration = time.time() - process_start
            return {
                "success": False,
                "error": str(e)
            }
    
    def delete_document(self, filename: str) -> bool:
        """Delete all chunks for a document"""
        return self.pinecone_service.delete_by_filename(filename)
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed documents"""
        return self.pinecone_service.get_stats()
