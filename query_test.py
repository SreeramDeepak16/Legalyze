from query_agent import QueryAgent

query = "Is disowning a child allowed?"

agent = QueryAgent()

result = agent.get_complete_query(query)

print(result)