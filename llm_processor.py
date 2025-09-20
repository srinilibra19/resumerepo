"""
LLM processing utilities for the Resume Chatbot application.
Handles model loading, caching, and query processing using sentence-transformers.
"""

import streamlit as st
import logging
from typing import Optional, Dict, Any, List
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
CHUNK_SIZE = 500  # Characters per chunk for document processing
OVERLAP_SIZE = 50  # Character overlap between chunks

class LLMProcessingError(Exception):
    """Custom exception for LLM processing errors"""
    pass

class ModelLoadingError(LLMProcessingError):
    """Exception raised when model loading fails"""
    pass

class QueryProcessingError(LLMProcessingError):
    """Exception raised when query processing fails"""
    pass

@st.cache_resource(show_spinner="Loading AI model...")
def load_sentence_transformer_model(model_name: str = DEFAULT_MODEL_NAME) -> Optional[SentenceTransformer]:
    """
    Load and cache the sentence transformer model with memory optimization.
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        Optional[SentenceTransformer]: Loaded model or None if loading fails
        
    Raises:
        ModelLoadingError: If model loading fails completely
    """
    try:
        logger.info(f"Loading sentence transformer model: {model_name}")
        
        # Load the model with memory optimization settings
        model = SentenceTransformer(
            model_name,
            device='cpu',  # Force CPU to save memory
            cache_folder=None  # Use default cache location
        )
        
        # Optimize model for inference (reduce memory usage)
        model.eval()
        
        logger.info(f"Successfully loaded model: {model_name}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        
        # Try fallback model if primary model fails
        if model_name != FALLBACK_MODEL_NAME:
            logger.info(f"Attempting to load fallback model: {FALLBACK_MODEL_NAME}")
            try:
                model = SentenceTransformer(
                    FALLBACK_MODEL_NAME,
                    device='cpu',
                    cache_folder=None
                )
                model.eval()
                logger.info(f"Successfully loaded fallback model: {FALLBACK_MODEL_NAME}")
                return model
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {str(fallback_error)}")
        
        raise ModelLoadingError(f"Failed to load any sentence transformer model: {str(e)}")

def get_model() -> SentenceTransformer:
    """
    Get the cached model instance with fallback handling.
    
    Returns:
        SentenceTransformer: The loaded model
        
    Raises:
        ModelLoadingError: If no model can be loaded
    """
    try:
        # Try to get the cached model
        model = load_sentence_transformer_model()
        
        if model is None:
            raise ModelLoadingError("Model loading returned None")
        
        return model
        
    except Exception as e:
        logger.error(f"Error getting model: {str(e)}")
        raise ModelLoadingError(f"Unable to load sentence transformer model: {str(e)}")

def initialize_model_in_session() -> bool:
    """
    Initialize the model and update session state with comprehensive error handling.
    
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    try:
        if not st.session_state.get("model_loaded", False):
            with st.spinner("Loading AI model..."):
                # Load the model (this will be cached)
                model = get_model()
                
                # Update session state
                st.session_state.model_loaded = True
                
                logger.info("Model successfully initialized in session")
                return True
        else:
            # Model already loaded
            return True
            
    except ModelLoadingError as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.session_state.model_loaded = False
        
        # Show detailed error to user
        st.error(f"❌ **AI Model Loading Failed**")
        st.error(f"**Error:** {str(e)}")
        st.info("💡 **Fallback:** The application will use basic keyword matching instead.")
        st.info("💡 **Note:** You can still ask questions about the resume, but responses may be less accurate.")
        
        return False
        
    except MemoryError:
        logger.error("Memory error during model loading")
        st.session_state.model_loaded = False
        
        st.error("❌ **Memory Limitation**")
        st.error("Not enough memory to load the AI model.")
        st.info("💡 **Solutions:**")
        st.info("• Close other browser tabs to free up memory")
        st.info("• Try refreshing the page")
        st.info("• The application will use basic keyword matching as fallback")
        
        return False
        
    except ConnectionError:
        logger.error("Connection error during model loading")
        st.session_state.model_loaded = False
        
        st.error("❌ **Connection Error**")
        st.error("Unable to download the AI model due to connection issues.")
        st.info("💡 **Solutions:**")
        st.info("• Check your internet connection")
        st.info("• Try again in a few moments")
        st.info("• The application will use basic keyword matching as fallback")
        
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error during model initialization: {str(e)}")
        st.session_state.model_loaded = False
        
        st.error(f"❌ **Unexpected Model Loading Error**")
        st.error(f"**Error:** {str(e)}")
        st.info("💡 **Fallback:** Using basic keyword matching instead.")
        st.info("💡 **Solutions:**")
        st.info("• Try refreshing the page")
        st.info("• Check your internet connection")
        st.info("• Contact support if the problem persists")
        
        return False

@st.cache_data
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP_SIZE) -> List[str]:
    """
    Split text into overlapping chunks for better processing with memory optimization.
    
    Args:
        text (str): Text to chunk
        chunk_size (int): Size of each chunk in characters
        overlap (int): Overlap between chunks in characters
        
    Returns:
        List[str]: List of text chunks
    """
    # Memory optimization: handle very large documents
    if len(text) > 500000:  # 500KB of text
        logger.warning(f"Large document detected ({len(text)} chars), using smaller chunks")
        chunk_size = min(chunk_size, 300)  # Reduce chunk size for large documents
        overlap = min(overlap, 30)
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this isn't the last chunk, try to break at a sentence or word boundary
        if end < len(text):
            # Look for sentence boundary (. ! ?)
            sentence_end = max(
                text.rfind('.', start, end),
                text.rfind('!', start, end),
                text.rfind('?', start, end)
            )
            
            if sentence_end > start:
                end = sentence_end + 1
            else:
                # Look for word boundary
                word_end = text.rfind(' ', start, end)
                if word_end > start:
                    end = word_end
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    # Memory optimization: limit total number of chunks
    if len(chunks) > 50:
        logger.warning(f"Too many chunks ({len(chunks)}), limiting to 50 most relevant")
        # Keep first 25 and last 25 chunks (beginning and end of document)
        chunks = chunks[:25] + chunks[-25:]
    
    return chunks

def test_model_functionality() -> Dict[str, Any]:
    """
    Test basic model functionality and return status information.
    
    Returns:
        Dict[str, Any]: Status information about model functionality
    """
    try:
        model = get_model()
        
        # Test encoding
        test_sentences = ["This is a test sentence.", "Another test sentence."]
        embeddings = model.encode(test_sentences)
        
        # Test similarity calculation
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return {
            "status": "success",
            "model_name": model._modules['0'].auto_model.config.name_or_path if hasattr(model, '_modules') else "unknown",
            "embedding_dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0]),
            "test_similarity": float(similarity),
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Model functionality test failed: {str(e)}")
        return {
            "status": "error",
            "model_name": None,
            "embedding_dimension": None,
            "test_similarity": None,
            "error": str(e)
        }

@st.cache_data
def encode_document_chunks(_model: SentenceTransformer, document_chunks: List[str]) -> np.ndarray:
    """
    Encode document chunks with caching for performance optimization.
    
    Args:
        _model (SentenceTransformer): Model instance (prefixed with _ to avoid hashing)
        document_chunks (List[str]): List of document text chunks
        
    Returns:
        np.ndarray: Encoded embeddings for all chunks
    """
    try:
        # Batch encode all chunks for efficiency
        embeddings = _model.encode(
            document_chunks,
            batch_size=32,  # Process in batches to manage memory
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings
    except Exception as e:
        logger.error(f"Error encoding document chunks: {str(e)}")
        raise QueryProcessingError(f"Failed to encode document chunks: {str(e)}")

def find_relevant_chunks(query: str, document_chunks: List[str], model: SentenceTransformer, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Find the most relevant document chunks for a given query using semantic similarity with caching.
    
    Args:
        query (str): User's question
        document_chunks (List[str]): List of document text chunks
        model (SentenceTransformer): Loaded sentence transformer model
        top_k (int): Number of top relevant chunks to return
        
    Returns:
        List[Dict[str, Any]]: List of relevant chunks with similarity scores
    """
    try:
        # Encode the query (not cached as queries are unique)
        query_embedding = model.encode([query], convert_to_numpy=True)
        
        # Encode document chunks with caching
        chunk_embeddings = encode_document_chunks(model, document_chunks)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                relevant_chunks.append({
                    'text': document_chunks[idx],
                    'similarity': float(similarities[idx]),
                    'index': int(idx)
                })
        
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error finding relevant chunks: {str(e)}")
        raise QueryProcessingError(f"Failed to find relevant document chunks: {str(e)}")

def create_context_prompt(query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    """
    Create a context-based prompt for answering the query.
    
    Args:
        query (str): User's question
        relevant_chunks (List[Dict[str, Any]]): Relevant document chunks
        
    Returns:
        str: Formatted context prompt
    """
    if not relevant_chunks:
        return f"Question: {query}\n\nContext: No relevant information found in the document.\n\nAnswer: I couldn't find information related to your question in the uploaded resume."
    
    # Combine relevant chunks into context
    context_parts = []
    for i, chunk in enumerate(relevant_chunks, 1):
        context_parts.append(f"Context {i} (similarity: {chunk['similarity']:.3f}):\n{chunk['text']}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Based on the following resume content, please answer the question. If the information is not available in the provided context, please say so clearly.

Question: {query}

Resume Content:
{context}

Answer: Based on the resume content provided"""
    
    return prompt

def generate_simple_response(query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    """
    Generate a simple response based on relevant chunks without using a generative model.
    This is a fallback method when advanced LLM is not available.
    
    Args:
        query (str): User's question
        relevant_chunks (List[Dict[str, Any]]): Relevant document chunks
        
    Returns:
        str: Generated response
    """
    if not relevant_chunks:
        return "I couldn't find information related to your question in the uploaded resume. Please try rephrasing your question or ask about different aspects of the candidate's background."
    
    # Extract key information from relevant chunks
    response_parts = []
    response_parts.append(f"Based on the resume content, here's what I found regarding your question:")
    response_parts.append("")
    
    for i, chunk in enumerate(relevant_chunks, 1):
        similarity_percentage = chunk['similarity'] * 100
        response_parts.append(f"**Relevant section {i}** (confidence: {similarity_percentage:.1f}%):")
        response_parts.append(chunk['text'])
        response_parts.append("")
    
    if len(relevant_chunks) == 1:
        response_parts.append("This appears to be the most relevant information from the resume for your question.")
    else:
        response_parts.append(f"I found {len(relevant_chunks)} relevant sections that might answer your question.")
    
    return "\n".join(response_parts)

def process_query(query: str, document_content: str) -> Dict[str, Any]:
    """
    Process a user query against the document content and return a response with comprehensive error handling.
    
    Args:
        query (str): User's question
        document_content (str): Full document text content
        
    Returns:
        Dict[str, Any]: Response with answer, confidence, and metadata
        
    Raises:
        QueryProcessingError: If query processing fails
    """
    try:
        # Validate inputs
        if not query or not query.strip():
            raise QueryProcessingError("Query cannot be empty")
        
        if not document_content or not document_content.strip():
            raise QueryProcessingError("Document content cannot be empty")
        
        # Clean and prepare query
        query = query.strip()
        
        # Check document size for memory management
        if len(document_content) > 1000000:  # 1MB of text
            logger.warning(f"Large document detected: {len(document_content)} characters")
        
        # Check if model is available
        try:
            model = get_model()
            model_available = True
        except ModelLoadingError:
            model_available = False
            logger.warning("Model not available, using fallback method")
        except MemoryError:
            model_available = False
            logger.warning("Memory limitation, using fallback method")
        
        # Chunk the document with error handling
        try:
            document_chunks = chunk_text(document_content)
            logger.info(f"Document split into {len(document_chunks)} chunks")
        except MemoryError:
            # If chunking fails due to memory, try smaller chunks
            logger.warning("Memory limitation during chunking, using smaller chunks")
            document_chunks = chunk_text(document_content, chunk_size=250, overlap=25)
        
        if model_available:
            try:
                # Use semantic similarity with the model
                relevant_chunks = find_relevant_chunks(query, document_chunks, model)
                
                # Generate response using simple method
                response_text = generate_simple_response(query, relevant_chunks)
                
                # Calculate overall confidence
                if relevant_chunks:
                    avg_confidence = sum(chunk['similarity'] for chunk in relevant_chunks) / len(relevant_chunks)
                else:
                    avg_confidence = 0.0
                
                model_used = "sentence-transformer"
                
            except MemoryError:
                logger.warning("Memory limitation during model processing, falling back to keyword matching")
                # Fallback to keyword matching
                relevant_chunks = find_relevant_chunks_fallback(query, document_chunks)
                response_text = generate_simple_response(query, relevant_chunks)
                avg_confidence = 0.5 if relevant_chunks else 0.0
                model_used = "keyword-fallback-memory"
                
            except Exception as e:
                logger.warning(f"Model processing failed, using fallback: {str(e)}")
                # Fallback to keyword matching
                relevant_chunks = find_relevant_chunks_fallback(query, document_chunks)
                response_text = generate_simple_response(query, relevant_chunks)
                avg_confidence = 0.3 if relevant_chunks else 0.0
                model_used = "keyword-fallback-error"
            
        else:
            # Fallback: simple keyword matching
            relevant_chunks = find_relevant_chunks_fallback(query, document_chunks)
            response_text = generate_simple_response(query, relevant_chunks)
            avg_confidence = 0.5 if relevant_chunks else 0.0
            model_used = "keyword-fallback"
        
        # Ensure we have a valid response
        if not response_text or not response_text.strip():
            response_text = "I apologize, but I couldn't generate a response to your question. Please try rephrasing your question or ask about different aspects of the resume."
            avg_confidence = 0.0
        
        return {
            "answer": response_text,
            "confidence": float(avg_confidence),
            "relevant_chunks_count": len(relevant_chunks),
            "total_chunks": len(document_chunks),
            "model_used": model_used,
            "query": query
        }
        
    except QueryProcessingError:
        # Re-raise our custom exceptions
        raise
    except MemoryError:
        logger.error("Memory error during query processing")
        raise QueryProcessingError("Memory limitation encountered while processing your question. Please try a shorter or simpler question.")
    except Exception as e:
        logger.error(f"Unexpected error processing query: {str(e)}")
        raise QueryProcessingError(f"Unexpected error during query processing: {str(e)}")

def find_relevant_chunks_fallback(query: str, document_chunks: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Fallback method to find relevant chunks using simple keyword matching.
    
    Args:
        query (str): User's question
        document_chunks (List[str]): List of document text chunks
        top_k (int): Number of top relevant chunks to return
        
    Returns:
        List[Dict[str, Any]]: List of relevant chunks with scores
    """
    try:
        # Extract keywords from query (simple approach)
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'where', 'when', 'why', 'how', 'who'}
        query_words = query_words - stop_words
        
        if not query_words:
            return []
        
        # Score each chunk based on keyword matches
        chunk_scores = []
        for i, chunk in enumerate(document_chunks):
            chunk_words = set(re.findall(r'\b\w+\b', chunk.lower()))
            
            # Calculate simple overlap score
            matches = query_words.intersection(chunk_words)
            score = len(matches) / len(query_words) if query_words else 0
            
            if score > 0:
                chunk_scores.append({
                    'text': chunk,
                    'similarity': score,
                    'index': i
                })
        
        # Sort by score and return top-k
        chunk_scores.sort(key=lambda x: x['similarity'], reverse=True)
        return chunk_scores[:top_k]
        
    except Exception as e:
        logger.error(f"Error in fallback chunk finding: {str(e)}")
        return []

def validate_query_input(query: str) -> bool:
    """
    Validate user query input.
    
    Args:
        query (str): User's question
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not query or not isinstance(query, str):
        return False
    
    query = query.strip()
    
    # Check minimum length
    if len(query) < 3:
        return False
    
    # Check maximum length (prevent abuse)
    if len(query) > 1000:
        return False
    
    return True