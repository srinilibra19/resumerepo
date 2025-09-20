"""
Simple tests for document processing functionality.
"""

import io
from document_processor import (
    extract_text_from_file, 
    validate_file_format, 
    validate_file_size,
    UnsupportedFileFormatError,
    FileSizeExceededError,
    CorruptedFileError
)

def test_validate_file_format():
    """Test file format validation"""
    print("Testing file format validation...")
    
    # Test valid formats
    try:
        assert validate_file_format("resume.txt") == True
        assert validate_file_format("resume.docx") == True
        print("✓ Valid formats accepted")
    except Exception as e:
        print(f"✗ Valid format test failed: {e}")
    
    # Test invalid formats
    try:
        validate_file_format("resume.pdf")
        print("✗ Invalid format test failed - should have raised exception")
    except UnsupportedFileFormatError:
        print("✓ Invalid format correctly rejected")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

def test_validate_file_size():
    """Test file size validation"""
    print("\nTesting file size validation...")
    
    # Test valid size
    try:
        assert validate_file_size(1024) == True  # 1KB
        print("✓ Valid file size accepted")
    except Exception as e:
        print(f"✗ Valid size test failed: {e}")
    
    # Test oversized file
    try:
        validate_file_size(15 * 1024 * 1024)  # 15MB
        print("✗ Oversized file test failed - should have raised exception")
    except FileSizeExceededError:
        print("✓ Oversized file correctly rejected")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

def test_extract_text_from_txt():
    """Test text extraction from .txt files"""
    print("\nTesting .txt file extraction...")
    
    # Create sample text content
    sample_text = "John Doe\nSoftware Engineer\nExperience: 5 years\nSkills: Python, JavaScript"
    file_content = sample_text.encode('utf-8')
    
    try:
        extracted_text, file_format = extract_text_from_file(file_content, "resume.txt")
        assert file_format == ".txt"
        assert "John Doe" in extracted_text
        assert "Software Engineer" in extracted_text
        print("✓ Text file extraction successful")
    except Exception as e:
        print(f"✗ Text file extraction failed: {e}")

def test_error_handling():
    """Test error handling for corrupted files"""
    print("\nTesting error handling...")
    
    # Test empty file
    try:
        extract_text_from_file(b"", "empty.txt")
        print("✗ Empty file test failed - should have raised exception")
    except CorruptedFileError:
        print("✓ Empty file correctly handled")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Test file with insufficient content
    try:
        extract_text_from_file(b"Hi", "short.txt")
        print("✗ Short file test failed - should have raised exception")
    except CorruptedFileError:
        print("✓ Short file correctly handled")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

if __name__ == "__main__":
    print("Running document processor tests...\n")
    
    test_validate_file_format()
    test_validate_file_size()
    test_extract_text_from_txt()
    test_error_handling()
    
    print("\nTest execution completed!")