"""RAG (Retrieval-Augmented Generation) service for document retrieval"""
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.core.config import settings
from app.core.logging_config import get_logger
from app.utils.dependencies import get_vector_store, get_openai_client
from app.utils.prompts import get_query_generation_prompt, build_rag_context
import json

logger = get_logger(__name__)


class RAGService:
    """Service for retrieving relevant documents from vector database"""
    
    def __init__(self):
        self.vector_store = get_vector_store()
        self.retriever = None
        self._init_retriever()
    
    def _init_retriever(self):
        """Initialize retriever once and cache it"""
        try:
            if self.vector_store:
                # Use similarity search with score_threshold for better filtering
                self.retriever = self.vector_store.as_retriever(
                    search_kwargs={
                        "k": settings.PINECONE_RAG_K,
                        "score_threshold": settings.PINECONE_RAG_SIMILARITY_THRESHOLD
                    }
                )
            else:
                self.retriever = None
        except Exception as e:
            logger.warning(f"Retriever initialization warning: {e}")
            # Fallback without score_threshold if not supported
            try:
                if self.vector_store:
                    self.retriever = self.vector_store.as_retriever(
                        search_kwargs={"k": settings.PINECONE_RAG_K}
                    )
                else:
                    self.retriever = None
            except Exception as e2:
                self.retriever = None
    
    def generate_search_queries(self, message: str) -> List[str]:
        """Generate multiple search queries using AI"""
        try:
            client = get_openai_client()
            
            response = client.chat.completions.create(
                model=settings.OPENAI_QUERY_GEN_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a search query optimization assistant. Generate diverse search queries to improve document retrieval."
                    },
                    {
                        "role": "user", 
                        "content": get_query_generation_prompt(message)
                    }
                ],
                temperature=settings.OPENAI_QUERY_GEN_TEMPERATURE,
                max_tokens=settings.OPENAI_QUERY_GEN_MAX_TOKENS
            )
            
            query_text = response.choices[0].message.content.strip()
            
            # Clean up response
            if query_text.startswith("```json"):
                query_text = query_text[7:]
            if query_text.startswith("```"):
                query_text = query_text[3:]
            if query_text.endswith("```"):
                query_text = query_text[:-3]
            query_text = query_text.strip()
            
            search_queries = json.loads(query_text)
            
            if isinstance(search_queries, list):
                valid_queries = [q.strip() for q in search_queries if isinstance(q, str) and len(q.strip()) > 2]
                if not valid_queries:
                    return [message]
                return valid_queries[:5]
            else:
                return [message]
                
        except json.JSONDecodeError as e:
            return [message]
        except Exception as e:
            logger.error(f"Error generating search queries with AI: {e}", exc_info=True)
            return [message]
    
    def search_documents(self, query: str) -> List[Any]:
        """Search for documents using a single query with similarity filtering"""
        if not self.retriever:
            return []
        
        try:
            docs = self.retriever.invoke(query)
            # Filter by similarity score if available
            return self._filter_by_similarity(docs)
        except:
            try:
                docs = self.retriever.get_relevant_documents(query)
                return self._filter_by_similarity(docs)
            except Exception as e:
                logger.warning(f"Document search error: {e}")
                return []
    
    def _filter_by_similarity(self, documents: List[Any]) -> List[Any]:
        """Filter documents by similarity score threshold"""
        if not documents:
            return []
        
        filtered_docs = []
        threshold = settings.PINECONE_RAG_SIMILARITY_THRESHOLD
        
        for doc in documents:
            # Check if document has similarity score in metadata
            if hasattr(doc, 'metadata') and doc.metadata:
                score = doc.metadata.get('score') or doc.metadata.get('similarity_score')
                if score is not None:
                    # Pinecone returns scores as 0-1, where 1 is most similar
                    if score >= threshold:
                        filtered_docs.append(doc)
                    continue
            
            # If no score available, include the document (let vector store handle filtering)
            # This ensures we don't lose documents when scores aren't available
            filtered_docs.append(doc)
        
        # If we filtered too aggressively, keep at least top results
        if not filtered_docs and documents:
            # Return top documents even if below threshold (might be edge case)
            return documents[:min(10, len(documents))]
        
        return filtered_docs
    
    def search_documents_parallel(self, queries: List[str]) -> List[Any]:
        """Search for documents using multiple queries in parallel"""
        if not self.retriever or not queries:
            return []
        
        relevant_docs = []
        
        if settings.CHAT_PARALLEL_SEARCH and len(queries) > 1:
            def search_query(query: str):
                return self.search_documents(query)
            
            with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
                futures = {executor.submit(search_query, q): q for q in queries}
                for future in as_completed(futures):
                    try:
                        docs = future.result()
                        if docs:
                            relevant_docs.extend(docs)
                    except Exception as e:
                        logger.warning(f"Query search exception: {e}")
        else:
            for query in queries:
                docs = self.search_documents(query)
                if docs:
                    relevant_docs.extend(docs)
        
        return relevant_docs
    
    def deduplicate_documents(self, documents: List[Any]) -> List[Any]:
        """Remove duplicate documents while preserving relevance order"""
        if not documents:
            return []
        
        seen_content = set()
        unique_docs = []
        
        for doc in documents:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            # Use longer hash for better deduplication (first 500 chars)
            content_hash = hash(content[:500])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # Return up to configured K, preserving order (most relevant first)
        return unique_docs[:settings.PINECONE_RAG_K]
    
    def retrieve_context(self, message: str, use_query_generation: bool = True) -> Optional[str]:
        """Retrieve relevant context from documents for a message with enhanced retrieval"""
        if not self.retriever or not settings.CHAT_RAG_ENABLED:
            return None
        
        try:
            # Generate search queries for better retrieval
            if use_query_generation and settings.CHAT_QUERY_GEN_ENABLED:
                search_queries = self.generate_search_queries(message)
                # Always include original message as first query
                if message not in search_queries:
                    search_queries.insert(0, message)
            else:
                search_queries = [message]
            
            # Search documents with multiple queries
            relevant_docs = self.search_documents_parallel(search_queries)
            
            # If no results from parallel search, try original message directly
            if not relevant_docs:
                relevant_docs = self.search_documents(message)
            
            # Deduplicate while preserving relevance order
            relevant_docs = self.deduplicate_documents(relevant_docs)
            
            # Log retrieval stats for debugging
            if relevant_docs:
                logger.info(f"Retrieved {len(relevant_docs)} document chunks for query: {message[:50]}...")
                return build_rag_context(relevant_docs)
            else:
                logger.warning(f"No relevant documents found for query: {message[:50]}...")
            
            return None
            
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}", exc_info=True)
            return None

