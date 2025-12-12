# test_search_agent.py
from dotenv import load_dotenv
load_dotenv()

from searchAgent import SearchAgent

# choose a query that matches the jurisdiction you want:
# For US law test: "Miranda v. Arizona 384 U.S. 436"
# For Indian law test: "limitation period to file complaint under negotiable instruments act"
# QUERY = "How does judicial review work in the U.S.?"

def run_all(Query):

    agent = SearchAgent()

    # 1) Mirix
    # mirix_hits = agent.search_mirix_paragraph(Query, top_k=3) # type: ignore
    # print("\n=== Mirix Results ===")
    # if not mirix_hits:
    #     print("No mirix hits")
    # for i, h in enumerate(mirix_hits, start=1):
    #     print(i, h.get("backend"), h.get("source"))
    #     # print("Snippet:", h.get("snippet") or "(no snippet)")
    #     print("---")

    # 2) CourtListener
    cl_hits = agent.search_courtlistener_paragraph(Query, top_k=1)
    # print("\n=== CourtListener Results ===")
    # if not cl_hits:
    #     print("No courtlistener hits")
    # for i, h in enumerate(cl_hits, start=1):
    #     print(i, h.get("backend"), h.get("source"))
    #     print("Title:", h.get("title"))
    #     # print("Snippet:", h.get("snippet") or "(no snippet)")
    #     print("---")

    # 3) Web
    web_hits = agent.search_web_paragraph(Query, top_k=1)
    # print("\n=== Web Results ===")
    # if not web_hits:
    #     print("No web hits (SERPER_API_KEY missing or request failed)")
    # for i, h in enumerate(web_hits, start=1):
    #     print(i, h.get("backend"), h.get("source"))
    #     print("Title:", h.get("title"))
    #     print("Snippet:", h.get("snippet") or "(no snippet)")
    #     print("---")
    
    result = []
    # result.append(mirix_hits)
    if cl_hits != []:
        result.append(cl_hits)

    result.append(web_hits)
    return result


# print(run_all("What are the laws about guns in texas"))
