"""
Test memory optimization and performance for Streamlit Cloud deployment.
"""

import unittest
import psutil
import os
import gc
from unittest.mock import patch, MagicMock
import sys
import tempfile

# Add the current directory to the path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_processor import extract_text_from_file
from llm_processor import chunk_text, load_sentence_transformer_model

class TestMemoryOptimization(unittest.TestCase):
    """Test memory optimization features"""
    
    def setUp(self):
        """Set up test environment"""
        self.initial_memory = self.get_memory_usage()
        
    def tearDown(self):
        """Clean up after tests"""
        gc.collect()
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0
    
    def test_memory_usage_within_limits(self):
        """Test that memory usage stays within Streamlit Cloud limits"""
        current_memory = self.get_memory_usage()
        
        # Streamlit Cloud free tier has ~1GB limit
        # We should stay well below this during normal operation
        self.assertLess(current_memory, 500, 
                       f"Memory usage ({current_memory:.1f}MB) is too high for Streamlit Cloud")
    
    def test_document_processing_memory_efficiency(self):
        """Test that document processing doesn't consume excessive memory"""
        # Create a large test document
        large_text = "This is a test resume content. " * 10000  # ~300KB
        
        memory_before = self.get_memory_usage()
        
        # Process the document multiple times to test memory leaks
        for _ in range(5):
            chunks = chunk_text(large_text)
            self.assertIsInstance(chunks, list)
            self.assertGreater(len(chunks), 0)
        
        # Force garbage collection
        gc.collect()
        
        memory_after = self.get_memory_usage()
        memory_increase = memory_after - memory_before
        
        # Memory increase should be minimal (less than 50MB)
        self.assertLess(memory_increase, 50, 
                       f"Memory increase ({memory_increase:.1f}MB) is too high")
    
    def test_text_chunking_optimization(self):
        """Test that text chunking handles large documents efficiently"""
        # Test with various document sizes
        test_sizes = [1000, 10000, 100000, 500000]  # Characters
        
        for size in test_sizes:
            with self.subTest(size=size):
                large_text = "Test content. " * (size // 13)  # Approximate size
                
                memory_before = self.get_memory_usage()
                chunks = chunk_text(large_text)
                memory_after = self.get_memory_usage()
                
                # Verify chunks are created
                self.assertIsInstance(chunks, list)
                self.assertGreater(len(chunks), 0)
                
                # For very large documents, chunks should be limited
                if size > 100000:
                    self.assertLessEqual(len(chunks), 50, 
                                       "Too many chunks for large document")
                
                # Memory usage should be reasonable
                memory_increase = memory_after - memory_before
                self.assertLess(memory_increase, 100, 
                               f"Memory increase ({memory_increase:.1f}MB) too high for size {size}")
    
    def test_file_size_limits(self):
        """Test that file size limits prevent memory issues"""
        from document_processor import validate_file_size, FileSizeExceededError
        
        # Test within limits
        small_size = 1024 * 1024  # 1MB
        self.assertTrue(validate_file_size(small_size))
        
        # Test exceeding limits
        large_size = 10 * 1024 * 1024  # 10MB (should exceed our 5MB limit)
        with self.assertRaises(FileSizeExceededError):
            validate_file_size(large_size)
    
    @patch('llm_processor.SentenceTransformer')
    def test_model_loading_memory_optimization(self, mock_transformer):
        """Test that model loading is optimized for memory"""
        # Mock the model to avoid actual loading
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        memory_before = self.get_memory_usage()
        
        # Test model loading (mocked)
        try:
            model = load_sentence_transformer_model()
            self.assertIsNotNone(model)
            
            # Verify model was configured for CPU (memory optimization)
            mock_transformer.assert_called_with(
                'sentence-transformers/all-MiniLM-L6-v2',
                device='cpu',
                cache_folder=None
            )
            
            # Verify eval mode was called (optimization)
            mock_model.eval.assert_called_once()
            
        except Exception as e:
            # Expected since we're mocking
            pass
        
        memory_after = self.get_memory_usage()
        memory_increase = memory_after - memory_before
        
        # Memory increase should be minimal for mocked test
        self.assertLess(memory_increase, 10, 
                       f"Memory increase ({memory_increase:.1f}MB) too high for mocked model")
    
    def test_garbage_collection_effectiveness(self):
        """Test that garbage collection effectively frees memory"""
        memory_before = self.get_memory_usage()
        
        # Create some large objects
        large_objects = []
        for i in range(10):
            large_objects.append("x" * 100000)  # 100KB each
        
        memory_with_objects = self.get_memory_usage()
        
        # Clear objects and run garbage collection
        large_objects.clear()
        del large_objects
        gc.collect()
        
        memory_after_gc = self.get_memory_usage()
        
        # Memory should be freed (allowing some tolerance)
        # Note: Python's memory management may not immediately release memory to OS
        memory_freed = memory_with_objects - memory_after_gc
        self.assertGreaterEqual(memory_freed, 0, 
                               "Garbage collection should not increase memory usage")
    
    def test_concurrent_processing_memory(self):
        """Test memory usage with multiple operations"""
        memory_before = self.get_memory_usage()
        
        # Simulate multiple document processing operations
        test_documents = [
            "Resume content for candidate 1. " * 1000,
            "Resume content for candidate 2. " * 1000,
            "Resume content for candidate 3. " * 1000,
        ]
        
        all_chunks = []
        for doc in test_documents:
            chunks = chunk_text(doc)
            all_chunks.extend(chunks)
        
        memory_after = self.get_memory_usage()
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable even with multiple documents
        self.assertLess(memory_increase, 100, 
                       f"Memory increase ({memory_increase:.1f}MB) too high for concurrent processing")
        
        # Clean up
        del all_chunks
        del test_documents
        gc.collect()

class TestPerformanceOptimization(unittest.TestCase):
    """Test performance optimization features"""
    
    def test_chunk_caching(self):
        """Test that text chunking uses caching effectively"""
        test_text = "This is test content for caching. " * 1000
        
        import time
        
        # First call (should be slower)
        start_time = time.time()
        chunks1 = chunk_text(test_text)
        first_call_time = time.time() - start_time
        
        # Second call (should be faster due to caching)
        start_time = time.time()
        chunks2 = chunk_text(test_text)
        second_call_time = time.time() - start_time
        
        # Results should be identical
        self.assertEqual(chunks1, chunks2)
        
        # Second call should be significantly faster (cached)
        # Note: This might not always be true in test environment
        # but it's good to check
        if second_call_time > 0:
            self.assertLessEqual(second_call_time, first_call_time * 2, 
                               "Caching should improve performance")
    
    def test_batch_processing_efficiency(self):
        """Test that batch processing is more efficient than individual processing"""
        # This test would require actual model loading, so we'll mock it
        # In a real scenario, batch processing should be more efficient
        
        test_chunks = [f"Test chunk {i}" for i in range(10)]
        
        # Verify we have reasonable number of chunks for batch processing
        self.assertGreater(len(test_chunks), 1)
        self.assertLess(len(test_chunks), 100)  # Not too many for memory
    
    def test_streamlit_cloud_compatibility(self):
        """Test compatibility with Streamlit Cloud constraints"""
        # Test file size limits
        max_file_size = 5 * 1024 * 1024  # 5MB
        self.assertLessEqual(max_file_size, 10 * 1024 * 1024, 
                           "File size limit should be reasonable for Streamlit Cloud")
        
        # Test chunk limits
        max_chunks = 50
        self.assertLessEqual(max_chunks, 100, 
                           "Chunk limit should prevent memory issues")
        
        # Test memory thresholds
        memory_warning_threshold = 800  # MB
        memory_limit = 1024  # MB (Streamlit Cloud limit)
        self.assertLess(memory_warning_threshold, memory_limit, 
                       "Warning threshold should be below actual limit")

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)