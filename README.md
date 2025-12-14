Legalyze – Memory-Augmented Legal AI System

Legalyze is an intelligent legal information retrieval and reasoning system that combines multi-agent orchestration, long-term memory, and web-assisted legal search.
Unlike traditional RAG-based systems, Legalyze continuously learns, remembers, validates, and refines legal knowledge over time.

The system integrates:
1.Mirix Orchestration Engine
2.L-MARS Legal Reasoning Pipeline
3.Meta Memory Manager
4.JudgeAgent & SummaryAgent
5.FastAPI backend
6.Electron-based desktop interface


Key Features:
1. Memory-first legal search (avoids redundant web queries)
2.Persistent evolving memory with validation
3.JudgeAgent for accuracy filtering
4.SummaryAgent for user-facing explanations
5.Web-assisted retrieval using SerpAPI
6.LLM-powered reasoning using Google Gemini
7.Desktop interface using Electron


System Architecture (High Level)
User Query
   ↓
Mirix Orchestrator
   ↓
Meta Memory Manager (Check Existing Knowledge)
   ↓
L-MARS Pipeline
   ├── Retrieval Agent (Web / Memory)
   ├── Judge Agent (Validation)
   └── Summary Agent (User Output)
   ↓
Memory Update
   ↓
Final Legal Answer


Tech Stack:

Backend
Python 3.10+
FastAPI
LangChain
Google Generative AI (Gemini)
SerpAPI
ChromaDB
Meta Memory Manager

Frontend
Electron.js
HTML / CSS / JavaScript


Installation & Setup
1️.Clone the Repository
git clone https://github.com/your-username/legalyze.git
cd legalyze

2.Create Virtual Environment (Recommended)
python -m venv venv
Activate it:
Windows
venv\Scripts\activate

3.Install Backend Dependencies 
Pip install -r requirements.txt


Environment Variables Setup:
Create a .env file in the project root:
SERPAPI_API_KEY=your_serpapi_key_here
GOOGLE_API_KEY=your_google_genai_key_here
COURTLISTENER_API_KEY=your_courtlistener_key_here


Running the Electron App:
npm start
This launches the Legalyze desktop interface.


How Legalyze Works:
1.User submits a legal query
2.Mirix checks existing memory first
3.If needed, L-MARS retrieves web results
4.JudgeAgent validates correctness
5.SummaryAgent generates final response
6.Verified knowledge is stored back into memory
7.This creates a self-improving legal reasoning loop.


Future Enhancements:
1.Multi-language legal support
2.User controlled memory deletion & updates
3.Jurisdiction-specific legal reasoning
4.Fine-grained memory confidence scoring
5.User-controlled memory management
6.Cloud-based multi-user support


Conclusion
Legalyze demonstrates how memory-driven multi-agent systems can significantly improve legal AI by reducing repetition, improving grounding, and enabling long-term reasoning.
The system evolves with use, making it more reliable and context-aware over time.


Author:
Project Name: Legalyze
Domain: Legal AI · Multi-Agent Systems · Memory-Augmented LLMs
Built for: Academic & Research Use


