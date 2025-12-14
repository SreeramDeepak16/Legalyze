from dotenv import load_dotenv #type:ignore
load_dotenv()

from agents.search_agent import SearchAgent

# choose a query that matches the jurisdiction you want:
# For US law test: "Miranda v. Arizona 384 U.S. 436"
# For Indian law test: "limitation period to file complaint under negotiable instruments act"
# QUERY = "How does judicial review work in the U.S.?"

def run_all(Query, isFirstTime : bool):

    agent = SearchAgent()
    mirix_hits = []
    cl_hits = []
    web_hits = []
    if (isFirstTime):
        mirix_hits = agent.search_mirix_paragraph(Query, top_k=3) 
    else:
        cl_hits = agent.search_courtlistener_paragraph(Query, top_k=3)
        web_hits = agent.search_web_paragraph(Query, top_k=3)
    
    result = []

    if mirix_hits != []:
        result.append(mirix_hits)
    print(mirix_hits)

    if cl_hits != []:
        result.append(cl_hits)

    if web_hits != []:
        result.append(web_hits)

    return result
