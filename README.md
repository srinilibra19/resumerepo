# Resume Chatbot

A Streamlit web application that allows users to upload and chat with resume documents using a lightweight Large Language Model (LLM). Optimized for deployment on Streamlit Cloud's free tier.

## Features

- 📄 Upload resume documents (.txt, .docx formats)
- 💬 Interactive chat interface for querying resume content
- 🤖 Lightweight LLM integration with sentence-transformers
- 🎨 Professional UI with clear instructions and feedback
- ⚡ Optimized for Streamlit Cloud deployment (< 1GB memory usage)

## 🚀 Deployment on Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Step-by-Step Deployment

1. **Fork this repository** to your GitHub account
   - Click the "Fork" button on this repository
   - Clone your fork locally if you want to make changes
   - Ensure your repository is **public** (required for Streamlit Cloud free tier)

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app" or "Create app"
   - **Repository:** Select your forked repository
   - **Branch:** Choose `main` or `master` (default branch)
   - **Main file path:** Set to `app.py`
   - **App URL:** Choose a custom URL (optional)
   - Click "Deploy!"

3. **Monitor deployment process** (usually 2-5 minutes)
   - Watch the deployment logs in real-time
   - Streamlit Cloud will automatically:
     - Install Python dependencies from `requirements.txt`
     - Download the AI model (~90MB) on first run
     - Configure the app with `.streamlit/config.toml` settings
   - The app will be available at `https://[your-app-name].streamlit.app`

4. **Verify deployment success**
   - Visit your app URL
   - Test file upload with the provided `sample_resume.txt`
   - Load the AI model and ask a test question
   - Check that all features work as expected

### Deployment Configuration

The app includes optimized configuration files:
- `requirements.txt`: Pinned dependencies for stable deployment
- `.streamlit/config.toml`: Streamlit-specific settings for cloud deployment

### Troubleshooting Deployment

**Common Issues and Solutions:**

**🔴 Deployment Fails to Start**
- **Issue**: Repository not found or access denied
- **Solution**: Ensure repository is public and you have proper permissions
- **Check**: Verify all files are committed and pushed to GitHub

**🔴 Memory Errors During Deployment**
- **Issue**: `MemoryError` or app crashes during model loading
- **Solution**: The app is optimized for 1GB limit, but initial model download may spike memory
- **Fix**: Wait for deployment to complete; model will be cached after first load
- **Alternative**: Try deploying during off-peak hours when resources are more available

**🔴 Model Loading Failures**
- **Issue**: `ConnectionError` or timeout during model download
- **Solution**: First-time model download (~90MB) may timeout; the app will retry automatically
- **Fix**: Refresh the app after 2-3 minutes; model will be cached for future use
- **Check**: Verify internet connectivity and Hugging Face Hub access

**🔴 Dependency Installation Errors**
- **Issue**: `pip install` fails or package conflicts
- **Solution**: All versions are pinned in `requirements.txt` to prevent conflicts
- **Fix**: Check deployment logs for specific package errors
- **Alternative**: Fork the repo and update `requirements.txt` if needed

**🔴 File Upload Issues**
- **Issue**: File upload fails or times out
- **Solution**: Check file size (max 5MB) and format (.txt, .docx only)
- **Fix**: Use the provided `sample_resume.txt` for testing

**🔴 App Runs But Features Don't Work**
- **Issue**: Chat interface doesn't respond or gives errors
- **Solution**: Ensure AI model loaded successfully (check left panel)
- **Fix**: Try the "Load AI Model" button and wait for confirmation

**📋 Deployment Checklist:**
- [ ] Repository is public on GitHub
- [ ] All files committed and pushed
- [ ] `requirements.txt` includes all dependencies
- [ ] `.streamlit/config.toml` is present
- [ ] `app.py` is in the root directory
- [ ] No sensitive data or API keys in code

**📊 Monitoring Your Deployment:**
- Check Streamlit Cloud dashboard for app status
- Monitor resource usage (memory, CPU) in the dashboard
- Review deployment logs for errors or warnings
- Test all features after deployment completes

**🆘 Getting Help:**
- Check Streamlit Cloud documentation: https://docs.streamlit.io/streamlit-cloud
- Review deployment logs in your Streamlit Cloud dashboard
- Open an issue on GitHub with detailed error information
- Contact Streamlit support for platform-specific issues

## 💻 Local Development

### Setup
1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd resume-chatbot
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the app** at `http://localhost:8501`

### Development Notes
- Models are cached locally after first download
- Session state is used for document storage (no database required)
- Hot reloading is enabled for development

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Memory**: 1GB RAM (optimized for Streamlit Cloud free tier)
- **Storage**: ~500MB for model downloads
- **Internet**: Required for initial model download

### Supported File Formats
- **Text files**: `.txt`
- **Word documents**: `.docx`
- **File size limit**: 5MB (configurable in `.streamlit/config.toml`)

## 📁 Project Structure

```
resume-chatbot/
├── app.py                      # Main Streamlit application
├── document_processor.py       # Document text extraction utilities
├── llm_processor.py            # LLM integration and query processing
├── requirements.txt            # Python dependencies (pinned versions)
├── README.md                   # Project documentation
├── .streamlit/
│   └── config.toml            # Streamlit configuration for deployment
├── test_document_processor.py  # Unit tests for document processing
├── test_llm_processor.py       # Unit tests for LLM functionality
└── test_query_engine.py        # Integration tests
```

## 🎯 Usage Guide

### Basic Workflow
1. **Upload Document**: Use the file uploader to select a resume (.txt or .docx)
2. **Wait for Processing**: The app will extract and process the document text
3. **Start Chatting**: Ask questions about the candidate's profile
4. **Review Responses**: View AI-generated answers based on the resume content

### Example Questions
- "What is the candidate's work experience?"
- "What programming languages does this person know?"
- "Tell me about their education background"
- "What are their key skills?"
- "Does this candidate have experience with [specific technology]?"

### Tips for Best Results
- Upload clear, well-formatted resume documents
- Ask specific questions rather than very broad ones
- If information isn't found, try rephrasing your question
- The chat history is maintained during your session

## 🔧 Technical Details

### Architecture
- **Frontend**: Streamlit web framework
- **NLP Model**: sentence-transformers/all-MiniLM-L6-v2 (~90MB)
- **Document Processing**: python-docx for Word documents
- **Deployment**: Optimized for Streamlit Cloud free tier

### Performance Optimizations
- Model caching with `@st.cache_resource`
- Efficient text processing and chunking
- Memory-optimized model selection
- Session-based storage (no persistent database)

### Security Features
- File format validation
- Size limits on uploads
- No persistent storage of sensitive data
- XSRF protection enabled

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Support

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting-deployment) section
2. Review Streamlit Cloud logs for deployment issues
3. Open an issue on GitHub with detailed error information

---

**Note**: This application is optimized for Streamlit Cloud's free tier limitations. For production use with larger documents or higher traffic, consider upgrading to Streamlit Cloud's paid tiers or deploying on other platforms.