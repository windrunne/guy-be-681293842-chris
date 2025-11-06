"""Service for generating embeddings"""
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.core.config import settings
from app.core.logging_config import get_logger
from app.utils.dependencies import get_embeddings

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings in parallel batches"""
    
    def __init__(self):
        self.embeddings = get_embeddings()
        self.batch_size = settings.DOCUMENT_EMBEDDING_BATCH_SIZE
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts in parallel batches"""
        if not texts:
            return []
        
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batches.append((i // self.batch_size + 1, texts[i:i + self.batch_size]))
        
        def generate_batch_embeddings(batch_num, batch_texts):
            try:
                return batch_num, self.embeddings.embed_documents(batch_texts), None
            except Exception as e:
                return batch_num, None, str(e)
        
        max_workers = min(len(batches), settings.DOCUMENT_EMBEDDING_PARALLEL_WORKERS)
        all_embeddings = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(generate_batch_embeddings, batch_num, batch_texts): batch_num
                for batch_num, batch_texts in batches
            }
            
            batch_results = {}
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    result_batch_num, batch_embeddings, error = future.result()
                    if error:
                        raise Exception(f"Batch {result_batch_num} error: {error}")
                    batch_results[result_batch_num] = batch_embeddings
                except Exception as e:
                    raise
        
        # Combine results in order
        for batch_num in sorted(batch_results.keys()):
            all_embeddings.extend(batch_results[batch_num])
        
        return all_embeddings

