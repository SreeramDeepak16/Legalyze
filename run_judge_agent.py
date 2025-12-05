from judgeAgent import EvaluateAgent
from summary_agent import SummaryAgent


ea = EvaluateAgent()
sa = SummaryAgent()

judge_output = ea.run()

text = sa.summarize_from_judge(judge_output, judge_output.get("results")[0][0].get("query"))
print(text)

# print(judge_output)
