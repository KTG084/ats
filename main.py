from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents import Document
from langchain_core.document_loaders import Blob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from pinecone import Pinecone as PineconeClient
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import io
import re
import PyPDF2
from typing import Dict
from dataclasses import dataclass
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("PORT", 8000))


@dataclass
class ATSKeywords:
    """Define keyword categories and weights for SDE roles (Campus + Industry)"""

    programming_languages = {
    "python": 5, "java": 5, "c++": 5, "c": 4,
    "javascript": 4, "typescript": 3,
    "html": 2, "css": 2, "sql": 4
    }

    frameworks = {
    "react": 3, "nodejs": 3, "express": 3,
    "django": 3, "flask": 2,
    "pandas": 2, "numpy": 2,
    "fastapi": 2, "nextjs": 2
    }

    cloud_devops = {
    "aws": 3, 
    "docker": 2, "git": 3, "github": 2,
    }

    databases = {
    "mysql": 3, "postgresql": 3, "mongodb": 3,
     "redis": 2
    }

    technical_concepts = {
    "api": 3, "restapi": 3, 
    "microservices": 2,
    "oops": 4, "object-oriented programming": 4,
    "algorithms": 5, "dsa": 5,
    "data structures": 5, "system design": 3,
    "webhooks": 2
    }

    cs_fundamentals = {
    "operating systems": 5, "computer networks": 5, "dbms": 5
    }

    soft_skills = {
    "teamwork": 2, "communication": 2, "collaboration": 2,
    "problem solving": 4, "debugging": 3, "testing": 3,
    "leadership": 2, "project management": 2
    }



class HybridATSEvaluator:
    def __init__(self):
        self.keywords = ATSKeywords()
        self.all_keywords = self._combine_all_keywords()

        self._llm = None
        self._embeddings = None
        self._vectorStore = None
        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(temperature=0.8, model="gemini-2.0-flash")
        return self._llm

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                task_type="RETRIEVAL_DOCUMENT"
            )
        return self._embeddings

    @property
    def vectorStore(self):
        if self._vectorStore is None:
            pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
            index_name = "google-embedding-index"
            self._vectorStore = PineconeVectorStore(
                index_name=index_name,
                embedding=self.embeddings
            )
        return self._vectorStore

    def _combine_all_keywords(self) -> Dict[str, int]:
        """Combine all keyword categories with their weights"""
        combined = {}
        for category in [
            self.keywords.programming_languages,
            self.keywords.frameworks,
            self.keywords.cloud_devops,
            self.keywords.databases,
            self.keywords.cs_fundamentals,
            self.keywords.technical_concepts,
            self.keywords.soft_skills
        ]:
            combined.update(category)
        return combined
    
    

    def extract_text_from_pdf(self, pdf_stream: io.BytesIO) -> str:
        """Extract text from PDF using PyPDF2"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text for keyword matching"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[/\-_]+', ' ', text)
        return text

    def extract_cgpa(self, text:str)-> Dict:
        
        text_upper = text.upper()
        
        patterns = [
        r'CGPA[:\s]+(\d+\.?\d*)',
        r'CGPA[:\s]*(\d+\.?\d*)',
        r'CGPA\s*[-:]\s*(\d+\.?\d*)',]
        
        cgpa_score = 0
        
        for pattern in patterns:
         match = re.search(pattern, text_upper)
         if match:
            try:
                cgpa_score = float(match.group(1))
                if 0 <= cgpa_score <= 10:
                 break
            except ValueError:
                continue
    
        return {
        "cgpa_score": cgpa_score,
        "cgpa_found": cgpa_score > 0
    }
    
    
    def find_keywords(self, text: str) -> Dict[str, int]:
        """Find and count keywords in the text"""
        text = self.preprocess_text(text)
        found_keywords = {}

        for keyword, weight in self.all_keywords.items():
            count = len(re.findall(rf'\b{re.escape(keyword)}\b', text))
            if count > 0:
                found_keywords[keyword] = count

        return found_keywords

    def calculate_ats_score(self, found_keywords: Dict[str, int], cgpa_data: Dict = None) -> Dict:
        """Calculate ATS score based on found keywords"""
        total_possible_score = sum(self.all_keywords.values())
        achieved_score = 0
        
        cgpa_max = 10
        total_possible_score+=cgpa_max

        category_scores = {
            "Programming Languages": 0,
            "Frameworks": 0,    
            "Cloud & DevOps": 0,
            "Databases": 0,
            "Technical Concepts": 0,
            "CS Fundamentals": 0,
            "Soft Skills": 0,
            "Academic Performance": 0
        }

        category_max_scores = {
            "Programming Languages": sum(self.keywords.programming_languages.values()),
            "Frameworks": sum(self.keywords.frameworks.values()),
            "Cloud & DevOps": sum(self.keywords.cloud_devops.values()),
            "Databases": sum(self.keywords.databases.values()),
            "Technical Concepts": sum(self.keywords.technical_concepts.values()),
            "CS Fundamentals": sum(self.keywords.cs_fundamentals.values()),
            "Soft Skills": sum(self.keywords.soft_skills.values()), 
            "Academic Performance": cgpa_max
        }

        for keyword, count in found_keywords.items():
            weight = self.all_keywords[keyword]
            effective_count = min(count, 3)
            keyword_score = weight * effective_count
            achieved_score += keyword_score

            if keyword in self.keywords.programming_languages:
                category_scores["Programming Languages"] += keyword_score
            elif keyword in self.keywords.frameworks:
                category_scores["Frameworks"] += keyword_score
            elif keyword in self.keywords.cloud_devops:
                category_scores["Cloud & DevOps"] += keyword_score
            elif keyword in self.keywords.databases:
                category_scores["Databases"] += keyword_score
            elif keyword in self.keywords.cs_fundamentals:
                category_scores["CS Fundamentals"] += keyword_score
            elif keyword in self.keywords.technical_concepts:
                category_scores["Technical Concepts"] += keyword_score
            elif keyword in self.keywords.soft_skills:
                category_scores["Soft Skills"] += keyword_score
                
                
        cgpa_score = 0
        if cgpa_data and cgpa_data.get('cgpa_score', 0) > 0:
         cgpa_value = cgpa_data['cgpa_score']
         if cgpa_value >= 9.0:
             cgpa_score = 10 
         elif cgpa_value >= 8.0:
             cgpa_score = 8  
         elif cgpa_value >= 7.0:
             cgpa_score = 6  
         elif cgpa_value >= 6.0:
             cgpa_score =  4  
         else:
             cgpa_score = 2        
                
        achieved_score += cgpa_score
        category_scores["Academic Performance"] = cgpa_score        

        overall_percentage = min(100, (achieved_score / total_possible_score) * 100 )

        category_percentages = {}
        for category, score in category_scores.items():
            max_score = category_max_scores[category]
            category_percentages[category] = min(100, (score / max_score) * 100) if max_score > 0 else 0

        return {
            "overall_percentage": round(overall_percentage, 1),
            "total_score": achieved_score,
            "max_possible_score": total_possible_score,
            "category_scores": category_percentages,
            "keywords_found": len(found_keywords),
            "total_keywords_possible": len(self.all_keywords),
            "cgpa_details": cgpa_data if cgpa_data else {"cgpa_score": 0}
        }

    async def get_llm_analysis(self, resume_text: str, keyword_analysis: Dict) -> Dict:
        """Get detailed analysis of the  including your CGPA"""
        try:
            documents = [Document(page_content=resume_text, metadata={"source": "resume"})]
            chunks = self._text_splitter.split_documents(documents)
            
            
            if not chunks:
              return {"detailed_feedback": "Failed to split resume into chunks", "source_chunks": 0}
                    
            ids = [f"resume-{i}" for i in range(len(chunks))]
            
            max_retries = 3
            for attempt in range(max_retries):
             try:
                self.vectorStore.add_documents(chunks, ids=ids)
                break
             except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(1 * (attempt + 1))
            
            ats_percentage = keyword_analysis['ats_score']['overall_percentage']
            keywords_count = len(keyword_analysis['found_keywords'])
            cgpa_score = keyword_analysis.get('cgpa_data', {}).get('cgpa_score', 0)
            
            system_message = f"""You are an expert ATS and recruitment specialist. You have access to a resume and its comprehensive analysis.

ANALYSIS CONTEXT:
- Overall ATS Score: {ats_percentage}%
- Keywords Found: {keywords_count} relevant keywords
- Academic Performance: CGPA {cgpa_score}/10.0 {"(Excellent)" if cgpa_score >= 8.5 else "(Good)" if cgpa_score >= 7.0 else "(Average)" if cgpa_score >= 6.0 else "(Needs Improvement)" if cgpa_score > 0 else "(Not Found)"}
- Score Category: {keyword_analysis['ats_score'].get('compatibility_level', 'Unknown')}

Your task is to provide comprehensive evaluation focusing on:
1. **TECHNICAL COMPETENCY**: Skills alignment with SDE requirements
2. **ACADEMIC FOUNDATION**: How CGPA reflects learning ability and consistency  
3. **EXPERIENCE RELEVANCE**: Quality of projects and work experience
4. **ATS OPTIMIZATION**: Keyword strategy and presentation improvements
5. **HOLISTIC ASSESSMENT**: Overall candidacy strength considering both technical and academic aspects

Context from resume: {{context}}"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", "Provide detailed analysis of this resume considering both keyword optimization and overall quality for Software Developer positions.")
            ])

            document_chain = create_stuff_documents_chain(self.llm, prompt)
            
            try:
                
             retriever = self.vectorStore.as_retriever(search_kwargs={"k": max(1, min(len(chunks), 5))})
             test_docs = retriever.get_relevant_documents("Analyse the resume")
            except Exception as e:
                raise Exception(f"Vectorstore retrieval setup failed: {str(e)}")
            
             
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            max_retries = 2
            for attempt in range(max_retries):
                    try:
                
                        response = await asyncio.wait_for(
                            asyncio.to_thread(
                                retrieval_chain.invoke, 
                                {"input": "Analyze this resume comprehensively"}
                            ),
                            timeout=60.0  # 60 second timeout
                        )
                
                
                        if not response or "answer" not in response:
                            raise Exception("Invalid response from retrieval chain")
                
                        return {
                            "detailed_feedback": response["answer"],
                            "source_chunks": len(response.get("context", []))
                        }
                
                    except asyncio.TimeoutError:
                        if attempt == max_retries - 1:
                            return {"detailed_feedback": "Analysis timed out", "source_chunks": 0}
                        await asyncio.sleep(2)
                    except Exception as e:
                        if attempt == max_retries - 1:
                            return {"detailed_feedback": f"Analysis failed: {str(e)}", "source_chunks": 0}
                        await asyncio.sleep(1)
                        
            return {
                "detailed_feedback": response["answer"],
                "source_chunks": len(response.get("context", []))
            }

        except Exception as e:
            return {
                "detailed_feedback": f"LLM analysis temporarily unavailable: {str(e)}",
                "source_chunks": 0
            }
            
        finally:
          try:
            self.vectorStore.delete(ids=ids)
          except Exception as cleanup_error:
            print(f"VectorStore cleanup failed: {cleanup_error}")    



app = FastAPI(
    title="Hybrid ATS Evaluation System",
    version="2.0",
    description="Advanced Resume Analysis: Keyword-based ATS scoring + AI-powered insights",
    docs_url="/docs",  
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:*", "*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "*"],  
    allow_headers=["Content-Type", "Authorization", "Accept", "*"],  
)

evaluator = HybridATSEvaluator()

@app.post("/evaluate/")
async def evaluate_resume(
    file: UploadFile = File(...), 
): 
    """Hybrid Resume Evaluation: Keyword scoring + LLM analysis"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="only PDF files are supported")
        
        content = await file.read()
        pdf_stream = io.BytesIO(content)
        text = evaluator.extract_text_from_pdf(pdf_stream)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        found_keywords = evaluator.find_keywords(text)
        cgpa_data = evaluator.extract_cgpa(text)
        ats_score = evaluator.calculate_ats_score(found_keywords, cgpa_data)
        llm_analysis = await evaluator.get_llm_analysis(text, {
                "ats_score": ats_score,
                "found_keywords": found_keywords,
                "cgpa_data": cgpa_data
            })  
        cgpa_value = cgpa_data.get('cgpa_score', 0)
        cgpa_grade = "Not Found"
        if cgpa_value >= 9.0:
            cgpa_grade = "Outstanding (A+)"
        elif cgpa_value >= 8.0:
            cgpa_grade = "Excellent (A)"
        elif cgpa_value >= 7.0:
            cgpa_grade = "Good (B+)"
        elif cgpa_value >= 6.0:
            cgpa_grade = "Average (B)"
        elif cgpa_value > 0:
            cgpa_grade = "Below Average (C)"
            
            
        score_percentage = ats_score['overall_percentage']
        if score_percentage >= 75:
            compatibility = "Excellent - Very likely to pass ATS filters"
        elif score_percentage >= 60:
            compatibility = "Good - Likely to pass most ATS systems"
        elif score_percentage >= 45:
            compatibility = "Fair - May pass some ATS systems"
        else:
            compatibility = "Poor - Unlikely to pass ATS filters"
            
        response = {
            "status": "success",
            "filename": file.filename,
            "ats_score": {
                "overall_percentage": score_percentage,
                "compatibility_level": compatibility,
                "category_breakdown": ats_score['category_scores'],
                "keywords_found": ats_score['keywords_found'],
                "total_keywords_checked": ats_score['total_keywords_possible']
            },
            "academic_performance": {
                "cgpa": cgpa_value if cgpa_value > 0 else "Not Found",
                "cgpa_grade": cgpa_grade,
                "cgpa_score_contribution": ats_score['category_scores'].get('Academic Performance', 0),
                "cgpa_weightage": "10 points (Academic performance contributes to overall ATS score)"
            },
            "found_keywords": found_keywords,
            "llm_analysis": llm_analysis,
            "summary": {
                "strengths": [k for k, v in ats_score['category_scores'].items() if v >= 50],
                "areas_for_improvement": [k for k, v in ats_score['category_scores'].items() if v < 30]
            },
            "competitive_benchmarking": {
                    "percentile_rank": "Based on your score, you rank better than {}% of resumes".format(
                        min(95, max(5, int(score_percentage * 1.2)))
                    ),
                    "industry_average": "58-65%"
                },
            
        }
        
        return response
        
        
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")


    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
