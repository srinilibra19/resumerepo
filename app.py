import streamlit as st
import datetime
from dataclasses import dataclass
from typing import List, Optional
import io
import gc  # Garbage collection for memory optimization
import psutil  # Memory monitoring
import os
from document_processor import (
    extract_text_from_file,
    UnsupportedFileFormatError,
    FileSizeExceededError,
    CorruptedFileError
)
from llm_processor import (
    initialize_model_in_session,
    test_model_functionality,
    process_query,
    validate_query_input,
    ModelLoadingError,
    QueryProcessingError
)

# Configure Streamlit page settings with memory optimization
st.set_page_config(
    page_title="Resume Chatbot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Memory monitoring functions
@st.cache_data
def get_memory_usage() -> dict:
    """Get current memory usage statistics."""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'percent': process.memory_percent()
        }
    except:
        return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}

def optimize_memory():
    """Perform garbage collection to optimize memory usage."""
    gc.collect()
    
def check_memory_limits() -> bool:
    """Check if memory usage is approaching limits."""
    try:
        memory_info = get_memory_usage()
        # Streamlit Cloud free tier has ~1GB limit
        return memory_info['rss_mb'] > 800  # Alert at 800MB
    except:
        return False

# Data models
@dataclass
class Document:
    filename: str
    content: str
    upload_timestamp: datetime.datetime
    file_size: int

@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime.datetime

# Initialize session state
if "document" not in st.session_state:
    st.session_state.document = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "processing" not in st.session_state:
    st.session_state.processing = False
if "model_attempted" not in st.session_state:
    st.session_state.model_attempted = False
if "model_available" not in st.session_state:
    st.session_state.model_available = True

def handle_file_upload():
    """Handle file upload and processing with comprehensive error handling and memory optimization"""
    uploaded_file = st.file_uploader(
        "Choose a resume file",
        type=['txt', 'docx'],
        help="Upload a resume in .txt or .docx format (max 5MB for optimal performance)",
        key="resume_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Display file information
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {file_size_mb:.2f} MB")
            
            # Process the uploaded file with enhanced progress indication
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📄 Reading file...")
            progress_bar.progress(25)
            
            # Extract text from the uploaded file
            file_content = uploaded_file.getvalue()
            
            status_text.text("🔍 Extracting text content...")
            progress_bar.progress(50)
            
            extracted_text, file_format = extract_text_from_file(
                file_content, 
                uploaded_file.name
            )
            
            status_text.text("💾 Saving document...")
            progress_bar.progress(75)
            
            # Create Document object and store in session state
            st.session_state.document = Document(
                filename=uploaded_file.name,
                content=extracted_text,
                upload_timestamp=datetime.datetime.now(),
                file_size=len(file_content)
            )
            
            status_text.text("✅ Processing complete!")
            progress_bar.progress(100)
            
            # Clear progress indicators after a brief moment
            import time
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # Memory optimization: clean up file content from memory
            del file_content
            optimize_memory()
            
            # Show success message with better formatting
            st.success(f"✅ **Document processed successfully!**")
            
            # Display processing results in organized format
            col_result1, col_result2 = st.columns(2)
            with col_result1:
                st.metric("📄 Format", file_format.upper())
            with col_result2:
                st.metric("📊 Content", f"{len(extracted_text):,} chars")
                
                # Show preview of extracted text
                with st.expander("📄 Document Preview", expanded=False):
                    preview_text = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
                    st.text_area(
                        "Extracted text preview:",
                        value=preview_text,
                        height=200,
                        disabled=True
                    )
                
        except UnsupportedFileFormatError as e:
            st.error(f"❌ **Unsupported File Format**")
            st.error(f"**Error:** {str(e)}")
            st.info("💡 **Solution:** Please upload a file in .txt or .docx format.")
            
        except FileSizeExceededError as e:
            st.error(f"❌ **File Too Large**")
            st.error(f"**Error:** {str(e)}")
            st.info("💡 **Solution:** Please reduce the file size or split the content into smaller documents.")
            
        except CorruptedFileError as e:
            st.error(f"❌ **File Processing Error**")
            st.error(f"**Error:** {str(e)}")
            st.info("💡 **Solutions:**")
            st.info("• Try saving the document in a different format")
            st.info("• Check if the file is corrupted and re-create it")
            st.info("• For Word documents, try saving as .docx format")
            
        except MemoryError:
            st.error(f"❌ **Memory Limitation**")
            st.error("The document is too large to process with available memory.")
            st.info("💡 **Solutions:**")
            st.info("• Try uploading a smaller document")
            st.info("• Split large documents into sections")
            st.info("• Remove unnecessary content like images or formatting")
            
        except PermissionError:
            st.error(f"❌ **File Access Error**")
            st.error("Unable to access the uploaded file.")
            st.info("💡 **Solution:** Please try uploading the file again.")
            
        except Exception as e:
            st.error(f"❌ **Unexpected Error**")
            st.error(f"**Error:** {str(e)}")
            st.info("💡 **Solutions:**")
            st.info("• Try refreshing the page and uploading again")
            st.info("• Check if the file is valid and not corrupted")
            st.info("• Contact support if the problem persists")
            
            # Log the error for debugging
            import logging
            logging.error(f"Unexpected file upload error: {str(e)}", exc_info=True)
    
    return uploaded_file is not None

def display_document_status():
    """Display current document status with enhanced formatting"""
    if st.session_state.document:
        st.success("✅ **Document Ready!**")
        
        # Display document information in a more organized way
        doc = st.session_state.document
        
        # Create an info box with document details
        with st.container():
            st.markdown("**📋 Document Details:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("📄 Filename", doc.filename)
                st.metric("📊 Content Length", f"{len(doc.content):,} chars")
            
            with col2:
                st.metric("💾 File Size", f"{doc.file_size / 1024:.1f} KB")
                upload_time = doc.upload_timestamp.strftime('%H:%M:%S')
                st.metric("⏰ Uploaded", upload_time)
        
        # Show document quality indicators
        content_length = len(doc.content)
        if content_length < 500:
            st.warning("⚠️ **Short Document:** This resume might be too brief for optimal results.")
        elif content_length > 10000:
            st.info("📄 **Detailed Resume:** This comprehensive document should provide great results!")
        else:
            st.success("👍 **Good Length:** This resume size is optimal for analysis.")
        
        # Option to clear the document with confirmation
        if st.button("🗑️ Clear Document", type="secondary", help="Remove current document and start over"):
            st.session_state.document = None
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("**📤 Ready to Upload**")
        st.markdown("Upload a resume document to get started with AI-powered analysis.")

def display_model_status():
    """Display AI model status and loading with comprehensive error handling"""
    st.markdown("### 🤖 AI Model Status")
    
    # Check if we've attempted to load the model
    model_attempted = st.session_state.get("model_attempted", False)
    model_loaded = st.session_state.get("model_loaded", False)
    model_available = st.session_state.get("model_available", True)
    
    if model_loaded and model_available:
        st.success("✅ AI model loaded and ready!")
        
        # Show model test button
        if st.button("🧪 Test Model", help="Test model functionality"):
            try:
                with st.spinner("Testing model..."):
                    test_result = test_model_functionality()
                    
                    if test_result["status"] == "success":
                        st.success("✅ Model test passed!")
                        st.write(f"**Model:** {test_result['model_name']}")
                        st.write(f"**Embedding dimension:** {test_result['embedding_dimension']}")
                        st.write(f"**Test similarity score:** {test_result['test_similarity']:.4f}")
                    else:
                        st.error(f"❌ **Model Test Failed**")
                        st.error(f"**Error:** {test_result['error']}")
                        st.info("💡 **Solutions:**")
                        st.info("• Try reloading the model")
                        st.info("• Refresh the page and try again")
                        st.info("• The model may still work for basic queries")
                        
            except Exception as e:
                st.error(f"❌ **Model Test Error**")
                st.error(f"**Error:** {str(e)}")
                st.info("💡 **Solution:** The model may still work despite the test failure.")
    
    elif model_attempted and not model_available:
        # Fallback mode is active
        st.warning("⚠️ **Fallback Mode Active**")
        st.info("**Status:** Using keyword-based search instead of AI model")
        st.info("💡 **Note:** You can still ask questions! Responses will use keyword matching.")
        
        # Option to retry model loading
        if st.button("🔄 Retry AI Model Loading", help="Try loading the AI model again"):
            st.session_state.model_attempted = False
            st.session_state.model_loaded = False
            st.session_state.model_available = True
            st.rerun()
    
    else:
        # Model not attempted yet
        st.info("🤖 **AI Model Ready to Load**")
        st.markdown("**Options:**")
        st.markdown("• **Load AI Model:** Better accuracy with semantic understanding")
        st.markdown("• **Skip and Use Fallback:** Keyword-based search (works offline)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Load AI Model", type="primary"):
                st.session_state.model_attempted = True
                try:
                    with st.spinner("Loading AI model... This may take a moment."):
                        success = initialize_model_in_session()
                        if success:
                            st.success("✅ Model loaded successfully!")
                        else:
                            st.info("✅ **Fallback mode activated** - You can still use the chat!")
                        st.rerun()
                        
                except Exception as e:
                    st.session_state.model_available = False
                    st.info("✅ **Fallback mode activated** - You can still use the chat!")
                    st.rerun()
        
        with col2:
            if st.button("⚡ Use Fallback Mode", help="Skip AI model and use keyword search"):
                st.session_state.model_attempted = True
                st.session_state.model_loaded = False
                st.session_state.model_available = False
                st.info("✅ **Fallback mode activated** - You can start chatting!")
                st.rerun()

def display_chat_interface():
    """Display the enhanced chat interface with better UX"""
    # Chat management controls with better layout
    col_clear, col_count, col_status = st.columns([1, 2, 1])
    
    with col_clear:
        if st.button("🗑️ Clear Chat", help="Clear conversation history"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col_count:
        message_count = len(st.session_state.chat_history)
        if message_count > 0:
            st.caption(f"💬 {message_count} messages in conversation")
        else:
            st.caption("💬 Start a new conversation")
    
    with col_status:
        if st.session_state.get("processing", False):
            st.caption("🤔 Thinking...")
        else:
            st.caption("✅ Ready")
    
    # Display chat history with enhanced formatting
    chat_container = st.container()
    
    with chat_container:
        # Show enhanced welcome message if no chat history
        if not st.session_state.chat_history:
            with st.chat_message("assistant"):
                st.markdown("👋 **Welcome! I'm your Resume Assistant.**")
                st.markdown("I've analyzed the uploaded resume and I'm ready to answer your questions!")
                
                # Show current mode status
                model_loaded = st.session_state.get("model_loaded", False)
                model_available = st.session_state.get("model_available", True)
                
                if model_loaded and model_available:
                    st.success("🤖 **AI Mode Active:** Using advanced semantic understanding")
                elif st.session_state.get("model_attempted", False):
                    st.info("⚡ **Fallback Mode Active:** Using keyword-based search")
                else:
                    st.info("🔧 **Ready to Start:** Load AI model or use fallback mode in the left panel")
                
                # Show categorized example questions
                st.markdown("### 🎯 **Popular Questions to Get Started:**")
                
                col_q1, col_q2 = st.columns(2)
                
                with col_q1:
                    st.markdown("**🏢 Experience & Skills:**")
                    st.markdown("• What is their work experience?")
                    st.markdown("• What programming languages do they know?")
                    st.markdown("• What are their key technical skills?")
                    st.markdown("• Tell me about their most recent job")
                
                with col_q2:
                    st.markdown("**🎓 Education & Background:**")
                    st.markdown("• What is their educational background?")
                    st.markdown("• What certifications do they have?")
                    st.markdown("• How many years of experience do they have?")
                    st.markdown("• What projects have they worked on?")
                
                st.markdown("---")
                st.info("💡 **Tip:** Ask specific questions for the best results!")
        
        # Display chat history with better formatting
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message.role):
                # Add message numbering for better tracking
                if message.role == "user":
                    st.markdown(f"**Question #{(i//2)+1}:** {message.content}")
                else:
                    st.write(message.content)
                
                # Show timestamp with better formatting
                timestamp_str = message.timestamp.strftime("%H:%M:%S")
                st.caption(f"*{timestamp_str}*")
    
    # Enhanced chat input with better placeholder and status
    if st.session_state.get("processing", False):
        st.info("🤔 Processing your question... Please wait.")
        user_input = st.chat_input(
            placeholder="Please wait while I process your previous question...",
            disabled=True
        )
    else:
        # Show helpful placeholder based on conversation state
        if not st.session_state.chat_history:
            placeholder = "Ask your first question about the resume... (e.g., 'What programming languages do they know?')"
        else:
            placeholder = "Ask another question about the resume..."
        
        user_input = st.chat_input(
            placeholder=placeholder,
            disabled=False
        )
    
    # Process user input when submitted
    if user_input:
        handle_chat_input(user_input)
    
    # Add helpful tips at the bottom
    if st.session_state.chat_history:
        with st.expander("💡 Tips for Better Results", expanded=False):
            st.markdown("""
            **For more accurate answers:**
            - Be specific in your questions
            - Ask about one topic at a time
            - If you don't get the answer you want, try rephrasing
            - Use keywords that might appear in the resume
            
            **Example follow-up questions:**
            - "Can you be more specific about their Python experience?"
            - "What companies did they work for?"
            - "How long did they work at [company name]?"
            """)

def handle_chat_input(user_input: str):
    """Handle user chat input and generate AI response with comprehensive error handling"""
    # Validate input
    if not validate_query_input(user_input):
        st.error("❌ **Invalid Question**")
        st.info("💡 Please enter a question with at least 3 characters and no more than 1000 characters.")
        return
    
    # Check if document is available
    if not st.session_state.document or not st.session_state.document.content:
        st.error("❌ **No Document Available**")
        st.info("💡 Please upload a resume document first.")
        return
    
    # Add user message to chat history
    user_message = ChatMessage(
        role="user",
        content=user_input,
        timestamp=datetime.datetime.now()
    )
    st.session_state.chat_history.append(user_message)
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_input)
        timestamp_str = user_message.timestamp.strftime("%H:%M:%S")
        st.caption(f"*{timestamp_str}*")
    
    # Set processing state
    st.session_state.processing = True
    
    # Generate AI response with comprehensive error handling
    try:
        # Show loading indicator
        with st.chat_message("assistant"):
            with st.spinner("Analyzing resume and generating response..."):
                # Process the query
                result = process_query(user_input, st.session_state.document.content)
            
            # Display the response with better formatting
            response_content = result["answer"]
            
            # Add confidence indicator at the top of response
            confidence = result['confidence']
            if confidence > 0.7:
                st.success("🎯 **High Confidence Response**")
            elif confidence > 0.4:
                st.warning("⚠️ **Medium Confidence Response**")
            elif confidence > 0.1:
                st.info("ℹ️ **Low Confidence Response**")
            else:
                st.error("❓ **No Relevant Information Found**")
            
            # Display the main response
            st.markdown(response_content)
            
            # Show processing details in an expander with better organization
            with st.expander("📊 **Response Analytics**", expanded=False):
                # Create metrics for better visualization
                col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                
                with col_metrics1:
                    st.metric(
                        "🎯 Confidence", 
                        f"{confidence:.1%}",
                        help="How confident the AI is in this response"
                    )
                
                with col_metrics2:
                    st.metric(
                        "📄 Relevant Sections", 
                        result['relevant_chunks_count'],
                        help="Number of resume sections used for this answer"
                    )
                
                with col_metrics3:
                    st.metric(
                        "🤖 AI Model", 
                        result['model_used'].replace('-', ' ').title(),
                        help="Which AI model was used for processing"
                    )
                
                # Additional details
                st.markdown("---")
                st.markdown("**📈 Processing Details:**")
                st.write(f"• **Total document sections analyzed:** {result['total_chunks']}")
                st.write(f"• **Query processed:** {len(result['query'])} characters")
                
                # Confidence interpretation
                st.markdown("**🎯 Confidence Scale:**")
                if confidence > 0.7:
                    st.success("**Excellent** - Very relevant information found in the resume")
                elif confidence > 0.4:
                    st.warning("**Good** - Somewhat relevant information found")
                elif confidence > 0.1:
                    st.info("**Fair** - Limited relevant information available")
                else:
                    st.error("**Poor** - No directly relevant information found")
            
            # Add assistant message to chat history
            assistant_message = ChatMessage(
                role="assistant",
                content=response_content,
                timestamp=datetime.datetime.now()
            )
            st.session_state.chat_history.append(assistant_message)
            
            # Show timestamp
            timestamp_str = assistant_message.timestamp.strftime("%H:%M:%S")
            st.caption(f"*{timestamp_str}*")
    
    except QueryProcessingError as e:
        # Handle query processing errors
        with st.chat_message("assistant"):
            st.error("❌ **Query Processing Error**")
            st.error(f"**Error:** {str(e)}")
            st.info("💡 **Solutions:**")
            st.info("• Try rephrasing your question")
            st.info("• Ask about different aspects of the resume")
            st.info("• Use simpler language in your question")
        
        # Add error message to chat history
        error_chat_message = ChatMessage(
            role="assistant",
            content=f"Query processing error: {str(e)}",
            timestamp=datetime.datetime.now()
        )
        st.session_state.chat_history.append(error_chat_message)
    
    except MemoryError:
        # Handle memory limitations
        with st.chat_message("assistant"):
            st.error("❌ **Memory Limitation**")
            st.error("The system ran out of memory while processing your question.")
            st.info("💡 **Solutions:**")
            st.info("• Try asking about specific sections of the resume")
            st.info("• Use shorter, more focused questions")
            st.info("• Refresh the page if the problem persists")
        
        # Add error message to chat history
        error_chat_message = ChatMessage(
            role="assistant",
            content="Memory limitation encountered while processing your question.",
            timestamp=datetime.datetime.now()
        )
        st.session_state.chat_history.append(error_chat_message)
    
    except TimeoutError:
        # Handle timeout errors
        with st.chat_message("assistant"):
            st.error("❌ **Processing Timeout**")
            st.error("The question took too long to process.")
            st.info("💡 **Solutions:**")
            st.info("• Try asking a simpler question")
            st.info("• Check your internet connection")
            st.info("• Try again in a moment")
        
        # Add error message to chat history
        error_chat_message = ChatMessage(
            role="assistant",
            content="Processing timeout - please try a simpler question.",
            timestamp=datetime.datetime.now()
        )
        st.session_state.chat_history.append(error_chat_message)
    
    except Exception as e:
        # Handle unexpected errors
        with st.chat_message("assistant"):
            st.error("❌ **Unexpected Error**")
            st.error(f"**Error:** {str(e)}")
            st.info("💡 **Solutions:**")
            st.info("• Try refreshing the page")
            st.info("• Try asking a different question")
            st.info("• Contact support if the problem persists")
        
        # Add error message to chat history
        error_chat_message = ChatMessage(
            role="assistant",
            content=f"Unexpected error: {str(e)}",
            timestamp=datetime.datetime.now()
        )
        st.session_state.chat_history.append(error_chat_message)
        
        # Log the error for debugging
        import logging
        logging.error(f"Unexpected chat error: {str(e)}", exc_info=True)
    
    finally:
        # Reset processing state
        st.session_state.processing = False
        
        # Memory optimization after processing
        optimize_memory()
        
        # Check memory limits and warn user if needed
        if check_memory_limits():
            st.warning("⚠️ Memory usage is high. Consider refreshing the page if you experience slowdowns.")
        
        # Rerun to update the interface
        st.rerun()

def main():
    """Main application function with enhanced UI/UX"""
    # Page header with instructions
    st.title("💼 Resume Chatbot")
    st.markdown("**Upload a resume and chat with it using AI!**")
    
    # Add helpful instructions at the top
    with st.expander("ℹ️ How to Use This Application", expanded=False):
        st.markdown("""
        **Step 1:** Upload a resume file (.txt or .docx format, max 10MB)
        
        **Step 2:** Load the AI model (this may take a moment)
        
        **Step 3:** Start asking questions about the resume!
        
        **Tips for better results:**
        - Ask specific questions about skills, experience, or education
        - Use clear, simple language
        - Try different phrasings if you don't get the answer you're looking for
        """)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📄 Document Upload")
        
        # Add upload instructions
        if not st.session_state.document:
            st.markdown("### 📤 Upload Resume")
            st.info("**Supported formats:** .txt, .docx (max 10MB)")
            st.markdown("**What makes a good resume for this tool:**")
            st.markdown("• Clear text formatting")
            st.markdown("• Well-organized sections")
            st.markdown("• Complete information about skills and experience")
        
        # Display current document status
        display_document_status()
        
        # File upload interface
        if not st.session_state.document:
            handle_file_upload()
        else:
            st.markdown("### 🔄 Upload New Resume")
            if st.button("📤 Upload Different Document", help="Clear current document and upload a new one"):
                st.session_state.document = None
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        
        # Display model status
        display_model_status()
        
        # Memory monitoring section (for development/debugging)
        if st.checkbox("🔧 Show Performance Info", help="Display memory usage and performance metrics"):
            st.markdown("### 📊 Performance Metrics")
            memory_info = get_memory_usage()
            
            col_mem1, col_mem2 = st.columns(2)
            with col_mem1:
                st.metric("Memory Usage", f"{memory_info['rss_mb']:.1f} MB")
            with col_mem2:
                st.metric("Memory %", f"{memory_info['percent']:.1f}%")
            
            # Memory warning
            if check_memory_limits():
                st.warning("⚠️ High memory usage detected. Consider uploading a smaller document.")
            else:
                st.success("✅ Memory usage is optimal")
            
            # Manual memory optimization button
            if st.button("🧹 Optimize Memory", help="Run garbage collection to free up memory"):
                optimize_memory()
                st.success("Memory optimization completed!")
                st.rerun()
        
        # Add usage tips
        if st.session_state.document and st.session_state.get("model_loaded", False):
            st.markdown("---")
            st.markdown("### 💡 Quick Tips")
            st.success("✅ Ready to chat!")
            st.markdown("**Try asking:**")
            st.markdown("• What programming languages does this person know?")
            st.markdown("• Tell me about their work experience")
            st.markdown("• What is their educational background?")
    
    with col2:
        st.header("💬 Chat with Resume")
        
        if st.session_state.document:
            # Show chat interface regardless of model status (fallback mode available)
            display_chat_interface()
        else:
            # No document uploaded yet
            st.info("**Getting Started:**")
            st.markdown("1. Upload a resume document in the left panel")
            st.markdown("2. Load the AI model")
            st.markdown("3. Start asking questions!")
            
            # Show example questions even when no document is loaded
            st.markdown("### 🎯 Example Questions You Can Ask")
            example_questions = [
                "What programming languages does the candidate know?",
                "Tell me about their work experience",
                "What is their educational background?",
                "What are their key skills and competencies?",
                "What projects have they worked on?",
                "How many years of experience do they have?",
                "What certifications do they have?",
                "What companies have they worked for?"
            ]
            
            for i, question in enumerate(example_questions, 1):
                st.markdown(f"**{i}.** {question}")
    
    # Enhanced footer with more information
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    
    with col_footer1:
        st.markdown("**🔒 Privacy**")
        st.caption("Documents are processed locally and not stored permanently")
    
    with col_footer2:
        st.markdown("**⚡ Performance**")
        st.caption("Optimized for Streamlit Cloud free tier")
    
    with col_footer3:
        st.markdown("**🤖 AI Model**")
        st.caption("Powered by sentence-transformers")

if __name__ == "__main__":
    main()
