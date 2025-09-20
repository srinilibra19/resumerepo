#!/usr/bin/env python3
"""
Deployment verification script for Resume Chatbot.
This script verifies that all dependencies and functionality work correctly
for Streamlit Cloud deployment.
"""

import sys
import os
import importlib
import subprocess
import tempfile
import gc
import psutil
from typing import Dict, List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

def check_required_packages() -> Dict[str, Tuple[bool, str]]:
    """Check if all required packages can be imported"""
    required_packages = {
        'streamlit': 'streamlit',
        'transformers': 'transformers',
        'sentence_transformers': 'sentence-transformers',
        'torch': 'torch',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'docx': 'python-docx',
        'sklearn': 'scikit-learn',
        'psutil': 'psutil',
        'requests': 'requests'
    }
    
    results = {}
    for package, pip_name in required_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            results[pip_name] = (True, f"✅ {version}")
        except ImportError as e:
            results[pip_name] = (False, f"❌ Not installed: {str(e)}")
        except Exception as e:
            results[pip_name] = (False, f"❌ Error: {str(e)}")
    
    return results

def check_memory_usage() -> Tuple[bool, str]:
    """Check current memory usage"""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb < 500:
            return True, f"✅ Memory usage: {memory_mb:.1f}MB (good for Streamlit Cloud)"
        elif memory_mb < 800:
            return True, f"⚠️ Memory usage: {memory_mb:.1f}MB (acceptable)"
        else:
            return False, f"❌ Memory usage: {memory_mb:.1f}MB (too high for Streamlit Cloud)"
    except Exception as e:
        return False, f"❌ Could not check memory: {str(e)}"

def check_file_structure() -> Dict[str, Tuple[bool, str]]:
    """Check if all required files are present"""
    required_files = {
        'app.py': 'Main application file',
        'requirements.txt': 'Python dependencies',
        'document_processor.py': 'Document processing module',
        'llm_processor.py': 'LLM processing module',
        '.streamlit/config.toml': 'Streamlit configuration',
        'README.md': 'Documentation',
        'sample_resume.txt': 'Example resume for testing'
    }
    
    results = {}
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            results[file_path] = (True, f"✅ {description} ({size_kb:.1f}KB)")
        else:
            results[file_path] = (False, f"❌ Missing: {description}")
    
    return results

def test_document_processing() -> Tuple[bool, str]:
    """Test document processing functionality"""
    try:
        from document_processor import extract_text_from_file, validate_file_format
        
        # Test file format validation
        assert validate_file_format("test.txt") == True
        assert validate_file_format("test.docx") == True
        
        # Test text extraction
        sample_text = "John Doe\nSoftware Engineer\nPython, JavaScript"
        file_content = sample_text.encode('utf-8')
        extracted_text, file_format = extract_text_from_file(file_content, "test.txt")
        
        assert file_format == ".txt"
        assert "John Doe" in extracted_text
        
        return True, "✅ Document processing works correctly"
        
    except Exception as e:
        return False, f"❌ Document processing failed: {str(e)}"

def test_llm_processing() -> Tuple[bool, str]:
    """Test LLM processing functionality (fallback mode)"""
    try:
        from llm_processor import process_query, validate_query_input, chunk_text
        
        # Test query validation
        assert validate_query_input("What is the candidate's experience?") == True
        assert validate_query_input("") == False
        
        # Test text chunking
        test_text = "This is a test document. " * 100
        chunks = chunk_text(test_text)
        assert len(chunks) > 0
        
        # Test query processing (will use fallback if model not available)
        document_content = "John Smith is a Python developer with 5 years of experience."
        result = process_query("What programming languages does John know?", document_content)
        
        assert 'query' in result
        assert 'answer' in result
        assert 'confidence' in result
        
        return True, "✅ LLM processing works correctly (fallback mode available)"
        
    except Exception as e:
        return False, f"❌ LLM processing failed: {str(e)}"

def test_streamlit_compatibility() -> Tuple[bool, str]:
    """Test Streamlit compatibility"""
    try:
        import streamlit as st
        
        # Check if we can import Streamlit components
        from streamlit import file_uploader, chat_message, chat_input
        
        # Check caching decorators
        from streamlit import cache_data, cache_resource
        
        return True, "✅ Streamlit compatibility verified"
        
    except Exception as e:
        return False, f"❌ Streamlit compatibility failed: {str(e)}"

def check_deployment_readiness() -> Tuple[bool, str]:
    """Check overall deployment readiness"""
    issues = []
    
    # Check for common deployment issues
    if not os.path.exists('.streamlit/config.toml'):
        issues.append("Missing Streamlit configuration")
    
    if not os.path.exists('requirements.txt'):
        issues.append("Missing requirements.txt")
    
    # Check requirements.txt content
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
            if 'streamlit' not in requirements:
                issues.append("Streamlit not in requirements.txt")
            if 'transformers' not in requirements:
                issues.append("Transformers not in requirements.txt")
    except:
        issues.append("Could not read requirements.txt")
    
    # Check for sensitive data
    sensitive_patterns = ['api_key', 'secret', 'password', 'token']
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        for pattern in sensitive_patterns:
                            if pattern in content and 'test' not in content:
                                issues.append(f"Potential sensitive data in {file}")
                                break
                except:
                    pass
    
    if issues:
        return False, f"❌ Deployment issues: {', '.join(issues)}"
    else:
        return True, "✅ Ready for deployment"

def run_comprehensive_verification():
    """Run all verification checks"""
    print("🚀 Resume Chatbot - Deployment Verification")
    print("=" * 50)
    
    # Python version check
    success, message = check_python_version()
    print(f"\n📋 Python Version: {message}")
    
    # Package checks
    print(f"\n📦 Package Dependencies:")
    package_results = check_required_packages()
    all_packages_ok = True
    for package, (success, message) in package_results.items():
        print(f"  {package}: {message}")
        if not success:
            all_packages_ok = False
    
    # Memory check
    print(f"\n💾 Memory Usage:")
    success, message = check_memory_usage()
    print(f"  {message}")
    
    # File structure check
    print(f"\n📁 File Structure:")
    file_results = check_file_structure()
    all_files_ok = True
    for file_path, (success, message) in file_results.items():
        print(f"  {file_path}: {message}")
        if not success:
            all_files_ok = False
    
    # Functionality tests
    print(f"\n🧪 Functionality Tests:")
    
    success, message = test_document_processing()
    print(f"  Document Processing: {message}")
    doc_processing_ok = success
    
    success, message = test_llm_processing()
    print(f"  LLM Processing: {message}")
    llm_processing_ok = success
    
    success, message = test_streamlit_compatibility()
    print(f"  Streamlit Compatibility: {message}")
    streamlit_ok = success
    
    # Deployment readiness
    print(f"\n🚀 Deployment Readiness:")
    success, message = check_deployment_readiness()
    print(f"  {message}")
    deployment_ready = success
    
    # Overall assessment
    print(f"\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    all_checks_passed = (
        all_packages_ok and 
        all_files_ok and 
        doc_processing_ok and 
        llm_processing_ok and 
        streamlit_ok and 
        deployment_ready
    )
    
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Your application is ready for Streamlit Cloud deployment.")
        print("\n📋 Next Steps:")
        print("1. Commit all changes to your GitHub repository")
        print("2. Ensure repository is public")
        print("3. Deploy to Streamlit Cloud at https://share.streamlit.io")
        print("4. Test the deployed application with sample_resume.txt")
        return True
    else:
        print("❌ SOME CHECKS FAILED!")
        print("⚠️ Please address the issues above before deploying.")
        print("\n🔧 Common Solutions:")
        print("• Install missing packages: pip install -r requirements.txt")
        print("• Check file paths and ensure all files are committed")
        print("• Review error messages for specific issues")
        return False

if __name__ == "__main__":
    success = run_comprehensive_verification()
    sys.exit(0 if success else 1)