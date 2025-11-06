"""Service for Pinecone vector database operations"""
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.core.config import settings
from app.core.logging_config import get_logger
from app.utils.dependencies import get_pinecone_client
import pinecone 

logger = get_logger(__name__)


class PineconeService:
    """Service for Pinecone vector database operations"""
    
    def __init__(self):
        self.pinecone_client = get_pinecone_client()
        self.batch_size = settings.PINECONE_BATCH_SIZE
    
    def get_index(self):
        """Get or create Pinecone index"""
        index_name = settings.PINECONE_INDEX_NAME
        
        if self.pinecone_client:
            try:
                return self.pinecone_client.Index(index_name)
            except Exception:
                try:
                    self.pinecone_client.create_index(
                        name=index_name,
                        dimension=1536,
                        metric="cosine"
                    )
                    return self.pinecone_client.Index(index_name)
                except Exception as e:
                    raise Exception(f"Failed to create/get index: {e}")
        else:
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine"
                )
            return pinecone.Index(index_name)
    
    def prepare_vectors(self, documents: List[Dict[str, Any]], embeddings: List[List[float]], filename: str) -> List[Dict[str, Any]]:
        """Prepare vectors for Pinecone upsert"""
        vectors = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vector_id = f"{filename}_{i}_{hash(doc['text'])}"
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": doc["text"],
                    **doc["metadata"]
                }
            })
        return vectors
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> int:
        """Upsert vectors to Pinecone in parallel batches"""
        if not vectors:
            return 0
        
        index = self.get_index()
        
        # Prepare batches
        vector_batches = []
        for i in range(0, len(vectors), self.batch_size):
            batch_vectors = vectors[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            vector_batches.append((batch_num, batch_vectors))
        
        total_batches = len(vector_batches)
        
        def upsert_batch(batch_num, batch_vectors):
            try:
                logger.debug(f"Upserting batch {batch_num}/{total_batches} ({len(batch_vectors)} vectors)")
                index.upsert(vectors=batch_vectors)
                return batch_num, len(batch_vectors), None
            except Exception as e:
                logger.error(f"Batch {batch_num} upsert failed: {str(e)}")
                return batch_num, 0, str(e)
        
        max_workers = min(total_batches, settings.DOCUMENT_PINECONE_UPSERT_PARALLEL_WORKERS)
        upserted_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(upsert_batch, batch_num, batch_vectors): batch_num
                for batch_num, batch_vectors in vector_batches
            }
            
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    result_batch_num, count, error = future.result()
                    if error:
                        raise Exception(f"Batch {result_batch_num} upsert error: {error}")
                    upserted_count += count
                except Exception as e:
                    logger.error(f"Failed to get result for batch {batch_num}: {str(e)}")
                    raise
        
        return upserted_count
    
    def delete_by_filename(self, filename: str) -> bool:
        """Delete all vectors for a document by filename"""
        try:
            index = self.get_index()
            index.delete(filter={"filename": filename})
            return True
        except Exception as e:
            logger.error(f"Error deleting document {filename}: {str(e)}", exc_info=True)
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            index = self.get_index()
            stats = index.describe_index_stats()
            return {
                "total_vectors": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 0),
                "index_fullness": stats.get("index_fullness", 0)
            }
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}", exc_info=True)
            return {"error": str(e)}

