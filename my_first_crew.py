from crewai import Agent, Task, Crew
from langchain_ollama import OllamaLLM

# Set up Ollama LLM
ollama_llm = OllamaLLM(model="ollama/llama3.2")

# Define your agents
researcher = Agent(
    role='Senior Research Analyst',
    goal='Discover new insights',
    backstory="""You're an expert at finding interesting information""",
    llm=ollama_llm,
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Write engaging content',
    backstory="""You're a talented writer who simplifies complex information""",
    llm=ollama_llm,
    verbose=True
)

# Create tasks
research_task = Task(
    description='Find interesting facts about AI in insurance',
    agent=researcher,
    expected_output='A list of interesting facts about AI in insurance'
)

write_task = Task(
    description='Write a short blog post about AI in insurance',
    agent=writer,
    expected_output='A short blog post about AI in insurance'
)

# Form the crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    verbose=True
)

# Execute the crew's tasks
result = crew.kickoff()

print("Here's the result:")
print(result)