#!/usr/bin/env python3
"""
Test script to demonstrate the query processing engine functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_processor import process_query

def test_query_processing_comprehensive():
    """Test the query processing engine with various scenarios"""
    
    # Sample resume content
    resume_content = """
    Jane Doe
    Senior Software Engineer
    Email: jane.doe@email.com
    Phone: (555) 123-4567
    
    PROFESSIONAL SUMMARY
    Experienced software engineer with 8 years in full-stack development.
    Specializes in Python, JavaScript, and cloud technologies.
    
    WORK EXPERIENCE
    
    Senior Software Engineer - TechCorp (2020-Present)
    • Led development of microservices architecture using Python and Docker
    • Implemented CI/CD pipelines with Jenkins and AWS
    • Mentored junior developers and conducted code reviews
    • Reduced system latency by 40% through optimization
    
    Software Engineer - StartupXYZ (2018-2020)
    • Developed React frontend applications
    • Built REST APIs using Node.js and Express
    • Worked with PostgreSQL and MongoDB databases
    • Collaborated with cross-functional teams in Agile environment
    
    Junior Developer - WebSolutions (2016-2018)
    • Created responsive websites using HTML, CSS, JavaScript
    • Maintained legacy PHP applications
    • Participated in client meetings and requirement gathering
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of Technology (2012-2016)
    GPA: 3.8/4.0
    
    SKILLS
    Programming Languages: Python, JavaScript, Java, PHP, SQL
    Frameworks: React, Node.js, Django, Flask
    Databases: PostgreSQL, MongoDB, MySQL
    Cloud: AWS, Docker, Kubernetes
    Tools: Git, Jenkins, JIRA, VS Code
    
    CERTIFICATIONS
    • AWS Certified Solutions Architect (2021)
    • Certified Scrum Master (2019)
    """
    
    # Test queries
    test_queries = [
        "What programming languages does Jane know?",
        "How many years of experience does she have?",
        "What is her current job title?",
        "Does she have any AWS experience?",
        "What is her educational background?",
        "Has she worked with databases?",
        "What about her experience with blockchain?",  # This should return "not found"
        "Tell me about her leadership experience"
    ]
    
    print("Testing Query Processing Engine")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)
        
        try:
            result = process_query(query, resume_content)
            
            print(f"Model Used: {result['model_used']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Relevant Chunks: {result['relevant_chunks_count']}")
            print(f"Answer:\n{result['answer']}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Query processing engine test completed!")

if __name__ == "__main__":
    test_query_processing_comprehensive()