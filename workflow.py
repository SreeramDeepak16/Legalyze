from agents.search_agent import SearchAgent
from agents.judge_agent import EvaluateAgent
from agents.summary_agent import SummaryAgent

search_agent = SearchAgent()
summary_agent = SummaryAgent()

def execute(query):

    evaluate_agent = EvaluateAgent(query)
    judge_output = evaluate_agent.run()
    text =  summary_agent.summarize_from_judge(judge_output)
    print(text)
    return text
    

    
