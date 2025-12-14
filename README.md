Legalyze – A Memory-Augmented Multi-Agent Framework for Adaptive Legal Reasoning

Legalyze is an intelligent legal information retrieval and reasoning system that combines multi-agent orchestration, long-term memory, and web-assisted legal search.<br>
Unlike traditional RAG-based systems, Legalyze continuously learns, remembers, validates, and refines legal knowledge over time.<br>

The system integrates:<br>
1. Mirix Orchestration Engine<br>
2. L-MARS Legal Reasoning Pipeline<br>
3. Meta Memory Manager<br>
4. JudgeAgent & SummaryAgent<br>
5. FastAPI backend<br>
6. Electron-based desktop interface<br>


Key Features:<br>
1. Memory-first legal search (avoids redundant web queries)<br>
2. Persistent evolving memory with validation<br>
3. JudgeAgent for accuracy filtering<br>
4. SummaryAgent for user-facing explanations<br>
5. Web-assisted retrieval using SerpAPI<br>
6. LLM-powered reasoning using Google Gemini<br>
7. Desktop interface using Electron<br>


Tech Stack:<br><br>
Backend<br>
 - Python 3.10+<br>
 - FastAPI<br>
 - LangChain<br>
 - Google Generative AI (Gemini)<br>
 - SerpAPI<br>
 - ChromaDB<br>
 - Meta Memory Manager<br>

Frontend<br>
 - Electron.js<br>
 - HTML / CSS / JavaScript<br>


Installation & Setup<br>
1️.Clone the Repository<br>
```git clone https://github.com/your-username/legalyze.git```<br>
```cd legalyze```

2.Create Virtual Environment (Recommended)<br>
```python -m venv venv```<br>
Activate it:<br>
Windows<br>
```venv\Scripts\activate```<br>

3.Install Backend Dependencies <br>
```pip install -r requirements.txt```<br>


Environment Variables Setup:<br>
Create a .env file in the project root:<br>
```SERPAPI_API_KEY=your_serpapi_key_here```<br>
```GOOGLE_API_KEY=your_google_genai_key_here```<br>
```COURTLISTENER_API_KEY=your_courtlistener_key_here```<br>


Running the Electron App:<br>
```npm install```<br>
```npm start```<br>
This launches the Legalyze desktop interface.<br>


How Legalyze Works:<br>
1. User submits a legal query<br>
2. Mirix checks existing memory first<br>
3. If needed, L-MARS retrieves web results<br>
4. JudgeAgent validates correctness<br>
5. SummaryAgent generates final response<br>
6. Verified knowledge is stored back into memory<br>
7. This creates a self-improving legal reasoning loop.<br>


Future Enhancements:<br>
1. Multi-language legal support<br>
2. User controlled memory deletion & updates<br>
3. Jurisdiction-specific legal reasoning<br>
4. Fine-grained memory confidence scoring<br>
5. User-controlled memory management<br>
6. Cloud-based multi-user support<br>


Conclusion<br>
Legalyze demonstrates how memory-driven multi-agent systems can significantly improve legal AI by reducing repetition, improving grounding, and enabling long-term reasoning.<br>
The system evolves with use, making it more reliable and context-aware over time.<br>


Project Name: Legalyze<br>
Domain: Legal AI · Multi-Agent Systems · Memory-Augmented LLMs<br>
Built for: Academic & Research Use


