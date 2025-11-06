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
    
    def generate_answer_focused_queries(self, message: str) -> List[str]:
        """Generate queries that directly search for potential answer terms"""
        try:
            client = get_openai_client()
            
            prompt = f"""Given this question: "{message}"

Think about what the SPECIFIC ANSWER might be in a document. Generate 5-8 search queries that directly target potential answer terms and concepts that would appear in the document's answer.

For example:
- Question: "What is the only good information about the stock?" 
- Answer queries: ["insider information", "insider knowledge", "insiders know", "insiders know better", "insider trading information", "what insiders know about stock"]

- Question: "What are the best technical indicators?"
- Answer queries: ["support resistance", "moving average", "MA 50", "MA 200", "divergence", "technical indicators support resistance", "best indicators moving average"]

Generate queries that search for the ANSWER TERMS, not just the question terms.

Return ONLY a JSON array of search query strings:
["query1", "query2", "query3", ...]"""
            
            response = client.chat.completions.create(
                model=settings.OPENAI_QUERY_GEN_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert at predicting document answers and generating search queries for those answers."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.5,  # Slightly higher for more creative answer predictions
                max_tokens=300
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
            
            answer_queries = json.loads(query_text)
            
            if isinstance(answer_queries, list):
                return [q.strip() for q in answer_queries if isinstance(q, str) and len(q.strip()) > 2]
            return []
                
        except Exception as e:
            logger.warning(f"Error generating answer-focused queries: {e}")
            return []
    
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
                # Return up to 12 queries for better semantic coverage
                return valid_queries[:12]
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
        """Filter documents by similarity score threshold (disabled by default for better recall)"""
        if not documents:
            return []
        
        # If threshold is 0, return all documents (no filtering)
        threshold = settings.PINECONE_RAG_SIMILARITY_THRESHOLD
        if threshold <= 0:
            return documents
        
        filtered_docs = []
        for doc in documents:
            # Check if document has similarity score in metadata
            if hasattr(doc, 'metadata') and doc.metadata:
                score = doc.metadata.get('score') or doc.metadata.get('similarity_score')
                if score is not None:
                    # Pinecone returns scores as 0-1, where 1 is most similar
                    if score >= threshold:
                        filtered_docs.append(doc)
                    continue
            
            # If no score available, include the document
            filtered_docs.append(doc)
        
        # If we filtered too aggressively, keep at least top results
        if not filtered_docs and documents:
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
            search_queries = []
            if use_query_generation and settings.CHAT_QUERY_GEN_ENABLED:
                # Generate question-based queries
                question_queries = self.generate_search_queries(message)
                search_queries.extend(question_queries)
                
                # Generate answer-focused queries (search for answer terms directly)
                answer_queries = self.generate_answer_focused_queries(message)
                search_queries.extend(answer_queries)
                
                # Always include original message as first query
                if message not in search_queries:
                    search_queries.insert(0, message)
                
                logger.info(f"Generated {len(question_queries)} question queries + {len(answer_queries)} answer queries = {len(search_queries)} total for: '{message[:50]}...'")
                logger.debug(f"Sample queries: {search_queries[:8]}...")
            else:
                search_queries = [message]
            
            # Search documents with multiple queries
            relevant_docs = self.search_documents_parallel(search_queries)
            
            # If no results from parallel search, try original message directly
            if not relevant_docs:
                logger.debug("No results from parallel search, trying direct search...")
                relevant_docs = self.search_documents(message)
            
            # If still no results, try direct vector store search as fallback
            if not relevant_docs and self.vector_store:
                logger.debug("Trying direct vector store search as fallback...")
                try:
                    # Use vector store's similarity_search_with_score for better control
                    results = self.vector_store.similarity_search_with_score(
                        message,
                        k=settings.PINECONE_RAG_K  # Get more results
                    )
                    # Include all results since threshold is disabled
                    relevant_docs = [doc for doc, score in results]
                    
                    if relevant_docs:
                        logger.info(f"Direct vector search fallback found {len(relevant_docs)} documents")
                except Exception as e:
                    logger.warning(f"Direct vector search fallback failed: {e}")
            
            # Deduplicate while preserving relevance order
            relevant_docs = self.deduplicate_documents(relevant_docs)
            
            # Log retrieval stats for debugging
            if relevant_docs:
                # Log sample of retrieved content for debugging
                sample_contents = []
                for i, doc in enumerate(relevant_docs[:3]):  # Log first 3 chunks
                    if hasattr(doc, 'page_content'):
                        content = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                        sample_contents.append(f"Chunk {i+1}: {content}")
                
                logger.info(f"Retrieved {len(relevant_docs)} document chunks for query: '{message[:50]}...'")
                logger.debug(f"Sample chunks:\n" + "\n".join(sample_contents))
                return build_rag_context(relevant_docs)
            else:
                logger.warning(f"No relevant documents found for query: '{message[:50]}...'")
            
            return None
            
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}", exc_info=True)
            return None

