from typing import List, Dict, Optional, Union
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from sentence_transformers import SentenceTransformer
import pinecone
from tqdm import tqdm
import torch
from google.generativeai import GenerativeModel
import google.generativeai as genai
import tiktoken
from redis import Redis
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and chunking"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
    def load_documents(self, source_dir: Union[str, Path]) -> List[Dict]:
        """
        Load documents from a directory
        """
        source_dir = Path(source_dir)
        documents = []
        
        # Configure loaders for different file types
        loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
        }
        
        for file_path in source_dir.rglob('*'):
            if file_path.suffix in loaders:
                try:
                    loader = loaders[file_path.suffix](str(file_path))
                    documents.extend(loader.load())
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    
        return documents
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split documents into chunks
        """
        chunks = []
        for doc in documents:
            try:
                doc_chunks = self.text_splitter.split_text(doc.page_content)
                chunks.extend([{
                    'content': chunk,
                    'metadata': doc.metadata
                } for chunk in doc_chunks])
            except Exception as e:
                logger.error(f"Error chunking document: {str(e)}")
        
        return chunks

class VectorStore:
    """Manages vector storage and retrieval using Pinecone"""
    
    def __init__(self, 
                 api_key: str,
                 environment: str,
                 index_name: str,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                index_name,
                dimension=self.embedding_model.get_sentence_embedding_dimension(),
                metric='cosine'
            )
            
        self.index = pinecone.Index(index_name)
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        return self.embedding_model.encode(texts)
    
    def upsert_documents(self, documents: List[Dict]):
        """
        Upload documents and their embeddings to Pinecone
        """
        batch_size = 100
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i + batch_size]
            texts = [doc['content'] for doc in batch]
            embeddings = self.embed_texts(texts)
            
            # Prepare vectors for upload
            vectors = []
            for j, (doc, embedding) in enumerate(zip(batch, embeddings)):
                vector_id = hashlib.md5(doc['content'].encode()).hexdigest()
                vectors.append((vector_id, embedding.tolist(), doc['metadata']))
                
            # Upload to Pinecone
            self.index.upsert(vectors=vectors)
    
    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve similar documents for a query
        """
        query_embedding = self.embed_texts([query])[0]
        results = self.index.query(query_embedding.tolist(), top_k=k, include_metadata=True)
        
        return [{
            'content': match.metadata['content'],
            'metadata': match.metadata,
            'score': match.score
        } for match in results.matches]

class ResponseCache:
    """Caches query responses using Redis"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, ttl: int = 3600):
        self.redis_client = Redis(host=host, port=port)
        self.ttl = ttl
        
    def get_cache_key(self, query: str) -> str:
        """Generate a cache key for a query"""
        return f"rag:response:{hashlib.md5(query.encode()).hexdigest()}"
    
    def get_cached_response(self, query: str) -> Optional[Dict]:
        """Retrieve cached response for a query"""
        cache_key = self.get_cache_key(query)
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        return None
    
    def cache_response(self, query: str, response: Dict):
        """Cache a response for a query"""
        cache_key = self.get_cache_key(query)
        self.redis_client.setex(
            cache_key,
            self.ttl,
            json.dumps(response)
        )

class RAGSystem:
    """Main RAG system that coordinates all components"""
    
    def __init__(self,
                 vector_store: VectorStore,
                 response_cache: ResponseCache,
                 google_api_key: str):
        
        self.vector_store = vector_store
        self.response_cache = response_cache
        genai.configure(api_key=google_api_key)
        self.model = GenerativeModel('gemini-pro')
        
    def generate_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """Generate a prompt for the LLM"""
        context = "\n\n".join([
            f"Content: {doc['content']}\nSource: {doc['metadata'].get('source', 'Unknown')}"
            for doc in context_docs
        ])
        
        return f"""Please answer the following question based on the provided context. 
        If you cannot answer the question based on the context, please say so.

        Context:
        {context}

        Question: {query}

        Answer:"""
    
    def generate_response(self, query: str, max_retries: int = 3) -> Dict:
        """
        Generate a response for a query using RAG
        """
        # Check cache first
        cached_response = self.response_cache.get_cached_response(query)
        if cached_response:
            logger.info("Cache hit for query")
            return cached_response
        
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(query)
        
        # Generate prompt
        prompt = self.generate_prompt(query, relevant_docs)
        
        # Get response from Gemini
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                
                result = {
                    'query': query,
                    'response': response.text,
                    'source_documents': relevant_docs,
                    'metadata': {
                        'model': "gemini-pro",
                        'timestamp': pd.Timestamp.now().isoformat()
                    }
                }
                
                # Cache the response
                self.response_cache.cache_response(query, result)
                
                return result
                
            except Exception as e:
                logger.error(f"Error generating response (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    raise
                
def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    doc_processor = DocumentProcessor()
    vector_store = VectorStore(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENVIRONMENT'),
        index_name='rag-documents'
    )
    response_cache = ResponseCache()
    
    # Initialize RAG system
    rag_system = RAGSystem(
        vector_store=vector_store,
        response_cache=response_cache,
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )
    
    # Example usage
    documents = doc_processor.load_documents('data/documents')
    chunks = doc_processor.chunk_documents(documents)
    vector_store.upsert_documents(chunks)
    
    # Example query
    query = "What are the main benefits of RAG systems?"
    response = rag_system.generate_response(query)
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()