from crewai import Agent, Task, Crew
from langchain_ollama import OllamaLLM

# Set up Ollama LLM
ollama_llm = OllamaLLM(model="mistral:latest", base_url="http://127.0.0.1:11434")

# Define the agents
market_researcher = Agent(
    role='Market Researcher',
    goal='Find trends in the job market for a specific role over the last 3 months',
    backstory="""You are an expert in analyzing job market trends. You scrape and analyze job postings and resumes to identify key skills and technologies for "solution architect" roles.""",
    llm=ollama_llm,
    verbose=True
)

career_advisor = Agent(
    role='Career Advisor',
    goal='Provide career advice based on market trends',
    backstory="""You are a seasoned career advisor who helps professionals advance their careers. You take market data and provide actionable advice.""",
    llm=ollama_llm,
    verbose=True
)

learning_planner = Agent(
    role='Learning Planner',
    goal='Create a learning plan based on career advice',
    backstory="""You are an expert in creating personalized learning plans. You take career goals and create a step-by-step learning path to achieve them.""",
    llm=ollama_llm,
    verbose=True
)

resume_updater = Agent(
    role='Resume Updater',
    goal='Update a resume with new skills and experiences',
    backstory="""You are a professional resume writer who can craft compelling resumes. You take a person's experience and skills and highlight them effectively.""",
    llm=ollama_llm,
    verbose=True
)

# Create tasks
research_task = Task(
    description='Search for "solution architect" resumes from the last 3 months and identify key trends.',
    agent=market_researcher,
    expected_output='A summary of the top 5-10 key skills and technologies found in "solution architect" resumes.'
)

advice_task = Task(
    description='Based on the market research, provide a summary of what to focus on to become a competitive solution architect.',
    agent=career_advisor,
    expected_output='A concise summary of the most important areas of focus for a solution architect.'
)

learning_plan_task = Task(
    description='Create a detailed learning plan based on the career advice.',
    agent=learning_planner,
    expected_output='A step-by-step learning plan with suggested resources.'
)

# Form the crew
career_crew = Crew(
    agents=[market_researcher, career_advisor, learning_planner],
    tasks=[research_task, advice_task, learning_plan_task],
    verbose=True
)

# Execute the crew's tasks
if __name__ == "__main__":
    result = career_crew.kickoff()
    print("Career Advisor Crew Final Result:")
    print(result)
