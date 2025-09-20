# Resume Chatbot - Testing Guide

This guide provides comprehensive testing scenarios to verify that the Resume Chatbot works correctly with various resume formats and question types.

## 🧪 Testing Scenarios

### 1. Document Upload Testing

#### Test Case 1.1: Text File Upload (.txt)
**File:** `sample_resume.txt` (provided)
**Expected Result:** 
- ✅ File uploads successfully
- ✅ Text extraction completes without errors
- ✅ Document preview shows extracted content
- ✅ File size and format metrics displayed correctly

**Steps:**
1. Open the application
2. Click "Choose a resume file"
3. Select `sample_resume.txt`
4. Verify successful processing message
5. Check document details panel

#### Test Case 1.2: Marketing Resume (.txt)
**File:** `sample_resume_marketing.txt` (provided)
**Expected Result:**
- ✅ Different industry resume processes correctly
- ✅ Marketing-specific terminology extracted
- ✅ Longer document handled efficiently

#### Test Case 1.3: Word Document Upload (.docx)
**File:** Create a simple .docx file with resume content
**Expected Result:**
- ✅ Word document processes successfully
- ✅ Formatting is stripped, plain text extracted
- ✅ Special characters handled correctly

#### Test Case 1.4: Invalid File Format
**File:** Any .pdf, .jpg, or other unsupported format
**Expected Result:**
- ❌ Upload rejected with clear error message
- ✅ Helpful guidance provided for supported formats

#### Test Case 1.5: Large File Test
**File:** Text file > 5MB
**Expected Result:**
- ❌ File rejected due to size limit
- ✅ Clear error message with size information

#### Test Case 1.6: Empty or Corrupted File
**File:** Empty .txt file or corrupted .docx
**Expected Result:**
- ❌ Processing fails with appropriate error
- ✅ User-friendly error message displayed

### 2. AI Model Loading Testing

#### Test Case 2.1: Model Loading Success
**Steps:**
1. Upload a resume successfully
2. Click "Load AI Model" button
3. Wait for model download/loading

**Expected Result:**
- ✅ Model loads successfully (may take 1-2 minutes first time)
- ✅ Success message displayed
- ✅ Chat interface becomes available

#### Test Case 2.2: Model Loading Failure (Network Issues)
**Scenario:** Simulate network connectivity issues
**Expected Result:**
- ❌ Model loading fails gracefully
- ✅ Fallback mode activated automatically
- ✅ Application remains functional with keyword search

#### Test Case 2.3: Model Test Functionality
**Steps:**
1. After model loads successfully
2. Click "Test Model" button

**Expected Result:**
- ✅ Model test completes successfully
- ✅ Model information displayed (name, dimensions, etc.)
- ✅ Test similarity score shown

### 3. Query Processing Testing

Use the provided `sample_resume.txt` (John Doe - Software Engineer) for these tests:

#### Test Case 3.1: Basic Information Queries
**Questions to test:**
- "What is the candidate's name?"
- "What is their current job title?"
- "Where is the candidate located?"
- "What is their email address?"

**Expected Results:**
- ✅ Accurate information extracted
- ✅ High confidence scores (>0.7)
- ✅ Clear, concise answers

#### Test Case 3.2: Technical Skills Queries
**Questions to test:**
- "What programming languages does John know?"
- "Does he have Python experience?"
- "What web technologies is he familiar with?"
- "What cloud platforms has he worked with?"
- "Tell me about his database experience"

**Expected Results:**
- ✅ Comprehensive skill lists provided
- ✅ Specific technologies mentioned
- ✅ Medium to high confidence scores

#### Test Case 3.3: Experience and Career Queries
**Questions to test:**
- "How many years of experience does he have?"
- "What companies has he worked for?"
- "What was his role at TechCorp?"
- "Tell me about his achievements"
- "What projects has he worked on?"

**Expected Results:**
- ✅ Career progression clearly explained
- ✅ Company names and roles identified
- ✅ Specific achievements highlighted

#### Test Case 3.4: Education and Certifications
**Questions to test:**
- "What is his educational background?"
- "Where did he go to university?"
- "What certifications does he have?"
- "When did he graduate?"

**Expected Results:**
- ✅ Education details provided accurately
- ✅ Certification information listed
- ✅ Dates and institutions mentioned

#### Test Case 3.5: Complex/Analytical Queries
**Questions to test:**
- "Is this candidate suitable for a senior Python developer role?"
- "What makes him qualified for cloud architecture?"
- "Compare his frontend vs backend experience"
- "What are his leadership qualities?"

**Expected Results:**
- ✅ Thoughtful analysis provided
- ✅ Evidence from resume cited
- ✅ Balanced assessment given

#### Test Case 3.6: Information Not Available
**Questions to test:**
- "What is his salary expectation?"
- "Does he have blockchain experience?"
- "What are his hobbies?"
- "Is he willing to relocate?"

**Expected Results:**
- ✅ Clear "information not available" response
- ✅ Low confidence scores (<0.2)
- ✅ Suggestion to rephrase or ask different questions

#### Test Case 3.7: Edge Case Queries
**Questions to test:**
- "" (empty query)
- "Hi" (too short)
- Very long query (>1000 characters)
- Special characters: "What about his C++ skills?"
- Numbers: "How many years at TechCorp?"

**Expected Results:**
- ✅ Input validation works correctly
- ✅ Appropriate error messages for invalid inputs
- ✅ Special characters handled properly

### 4. Marketing Resume Testing

Use `sample_resume_marketing.txt` (Sarah Johnson - Digital Marketing Manager):

#### Test Case 4.1: Industry-Specific Queries
**Questions to test:**
- "What marketing channels has she worked with?"
- "Does she have Google Ads experience?"
- "What is her experience with social media?"
- "Tell me about her campaign results"
- "What marketing tools does she use?"

**Expected Results:**
- ✅ Marketing-specific terminology recognized
- ✅ Campaign metrics and results highlighted
- ✅ Tool and platform experience listed

#### Test Case 4.2: Cross-Industry Comparison
**Steps:**
1. Test with technical resume (John Doe)
2. Clear chat and upload marketing resume (Sarah Johnson)
3. Ask similar questions about skills and experience

**Expected Results:**
- ✅ Different skill sets properly identified
- ✅ Industry-appropriate responses
- ✅ No confusion between different resumes

### 5. Performance and Memory Testing

#### Test Case 5.1: Memory Usage Monitoring
**Steps:**
1. Enable "Show Performance Info" checkbox
2. Monitor memory usage during various operations
3. Upload large documents and process queries

**Expected Results:**
- ✅ Memory usage stays below 800MB
- ✅ Memory optimization works effectively
- ✅ No memory leaks during extended use

#### Test Case 5.2: Concurrent Operations
**Steps:**
1. Upload document
2. Load model
3. Process multiple queries rapidly
4. Clear chat and repeat

**Expected Results:**
- ✅ Application remains responsive
- ✅ No crashes or freezing
- ✅ Consistent performance across operations

#### Test Case 5.3: Session State Management
**Steps:**
1. Upload document and chat
2. Refresh the page
3. Upload different document
4. Clear document and start over

**Expected Results:**
- ✅ Session state resets properly on refresh
- ✅ Document switching works correctly
- ✅ Chat history managed appropriately

### 6. User Interface Testing

#### Test Case 6.1: Responsive Design
**Steps:**
1. Test on different screen sizes
2. Resize browser window
3. Check mobile compatibility (if applicable)

**Expected Results:**
- ✅ Layout adapts to different screen sizes
- ✅ All buttons and inputs remain accessible
- ✅ Text remains readable

#### Test Case 6.2: Error Handling UI
**Steps:**
1. Trigger various error conditions
2. Check error message display
3. Verify recovery options

**Expected Results:**
- ✅ Error messages are user-friendly
- ✅ Clear instructions for resolution
- ✅ Application recovers gracefully

#### Test Case 6.3: Loading States
**Steps:**
1. Monitor loading indicators during:
   - File upload
   - Model loading
   - Query processing

**Expected Results:**
- ✅ Loading indicators appear appropriately
- ✅ Progress feedback provided
- ✅ User knows when operations complete

## 🚀 Deployment Testing

### Test Case 7.1: Streamlit Cloud Deployment
**Steps:**
1. Deploy to Streamlit Cloud
2. Wait for deployment to complete
3. Access the deployed URL
4. Run through key test scenarios

**Expected Results:**
- ✅ Deployment completes without errors
- ✅ All functionality works in cloud environment
- ✅ Model downloads successfully on first use
- ✅ Performance remains acceptable

### Test Case 7.2: First-Time User Experience
**Steps:**
1. Access deployed app with fresh browser
2. Follow the application flow as a new user
3. Use provided sample files

**Expected Results:**
- ✅ Instructions are clear and helpful
- ✅ Sample files work correctly
- ✅ User can complete full workflow without issues

## 📋 Testing Checklist

Before considering deployment complete, verify:

- [ ] Both sample resume files upload and process correctly
- [ ] AI model loads successfully (or fallback works)
- [ ] All question categories return appropriate responses
- [ ] Error handling works for invalid inputs
- [ ] Memory usage stays within acceptable limits
- [ ] UI is responsive and user-friendly
- [ ] Deployment verification script passes
- [ ] Application works in Streamlit Cloud environment

## 🐛 Common Issues and Solutions

**Issue:** Model fails to load
**Solution:** Check internet connectivity; fallback mode should activate automatically

**Issue:** Large documents cause memory errors
**Solution:** Use smaller documents or split content; check file size limits

**Issue:** Queries return "no information found"
**Solution:** Try rephrasing questions; ensure document contains relevant information

**Issue:** Slow response times
**Solution:** Check memory usage; consider refreshing the application

**Issue:** Upload fails
**Solution:** Verify file format (.txt, .docx) and size (<5MB)

## 📊 Success Criteria

The application passes testing if:
- ✅ 95%+ of basic queries return relevant information
- ✅ All file format validations work correctly
- ✅ Error handling provides helpful guidance
- ✅ Memory usage stays below 800MB during normal operation
- ✅ Application recovers gracefully from errors
- ✅ Deployment completes successfully on Streamlit Cloud