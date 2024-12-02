import os
from crewai import Agent, Task, Crew
from langchain_ollama import OllamaLLM
from litellm import completion

# Function to get direct Ollama response
def get_ollama_response(prompt, max_words=50):
    try:
        print(f"Attempting to connect to Ollama at http://localhost:11434...")
        response = completion(
            model="ollama/llama3",
            messages=[
                {"content": f"Respond in {max_words} words or less. {prompt}", "role": "user"}
            ],
            api_base="http://localhost:11434"
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Detailed Ollama error: {str(e)}")
        return f"Error in Ollama response: {str(e)}"

# Initialize the Ollama Llama3 language model
try:
    print("Initializing Ollama LLM...")
    llm = OllamaLLM(
        model="llama2",
        base_url="http://localhost:11434",
        temperature=0.7
    )
except Exception as e:
    print(f"Failed to initialize Ollama LLM: {str(e)}")
    exit(1)

# Agent 1: AI Agents Research Specialist
ai_agents_researcher = Agent(
    role='AI Agents Research Analyst',
    goal='Conduct deep, comprehensive research on the latest developments in AI agent technologies',
    backstory='''You are a world-leading expert in AI agent research, with a profound understanding 
    of multi-agent systems, autonomous AI technologies, and emerging computational paradigms. 
    Your mission is to uncover the most innovative and transformative developments in the field 
    of AI agents, exploring their technical foundations, potential applications, and future implications.''',
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# Agent 2: Technological Trend Forecaster
tech_trend_analyst = Agent(
    role='AI Technology Trend Forecaster',
    goal='Analyze and predict future trends and potential applications of AI agent technologies',
    backstory='''You are a strategic technology futurist specializing in AI and computational intelligence. 
    Your expertise lies in connecting current technological developments with potential future scenarios, 
    identifying how AI agents might transform industries, solve complex problems, and reshape human-machine interactions.''',
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# Agent 3: Ethical AI and Policy Investigator
ethics_policy_agent = Agent(
    role='AI Ethics and Policy Researcher',
    goal='Examine the ethical implications and policy considerations surrounding AI agent technologies',
    backstory='''You are a meticulous researcher focused on the ethical, social, and legal dimensions 
    of AI agent technologies. Your work involves deep analysis of potential risks, societal impacts, 
    and the development of responsible AI frameworks that ensure these powerful technologies 
    are developed and deployed with the highest ethical standards.''',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Research Task for AI Agents
ai_agents_research_task = Task(
    description='''Conduct an in-depth investigation into current AI agent technologies:
    - Detailed analysis of state-of-the-art multi-agent systems
    - Comprehensive review of key research institutions and their breakthrough work
    - Technical breakdown of agent architecture, communication protocols, and learning mechanisms
    - Exploration of cutting-edge frameworks and development tools
    - Identification of most promising research directions''',
    agent=ai_agents_researcher,
    expected_output='A comprehensive 20-page research report detailing the current landscape of AI agent technologies'
)

# Future Trends Task
tech_trends_task = Task(
    description='''Develop a forward-looking analysis of AI agent technologies:
    - Predict potential breakthrough applications in next 5-10 years
    - Identify industries most likely to be transformed by AI agents
    - Analyze potential economic and societal impacts
    - Explore emerging interdisciplinary applications
    - Create speculative scenarios of AI agent evolution''',
    agent=tech_trend_analyst,
    expected_output='A detailed predictive report with 10 key future scenarios for AI agent technologies'
)

# Ethics and Policy Task
ethics_policy_task = Task(
    description='''Produce a comprehensive examination of ethical and policy considerations:
    - Detailed risk assessment of AI agent technologies
    - Analysis of potential societal and economic disruptions
    - Recommendations for ethical development frameworks
    - Investigation of legal and regulatory challenges
    - Proposals for responsible AI agent governance''',
    agent=ethics_policy_agent,
    expected_output='A comprehensive policy white paper on ethical guidelines and governance for AI agent technologies'
)

# Create Crew and Execute
crew = Crew(
    agents=[ai_agents_researcher, tech_trend_analyst, ethics_policy_agent],
    tasks=[ai_agents_research_task, tech_trends_task, ethics_policy_task],
    verbose=True
)

# Update the crew execution with error handling
try:
    print("\nStarting crew execution...")
    result = crew.kickoff()
    print("\nCrew execution result:")
    print(result)
except Exception as e:
    print(f"\nError during crew execution: {str(e)}")

# Additional Ollama Direct Interaction Example
print("\n--- Direct Ollama Interaction Examples ---")

# Example 1: Quick Agent Identity Check
agent_identity = get_ollama_response("Briefly describe your role and capabilities as an AI research agent")
print("Agent Identity Insight:", agent_identity)

# Example 2: Trend Prediction Quick Probe
trend_probe = get_ollama_response("What are the top 3 emerging trends in AI agent technologies for the next 2 years?")
print("Trend Prediction Snapshot:", trend_probe)

# Example 3: Ethical Considerations Probe
ethics_probe = get_ollama_response("Summarize the key ethical concerns surrounding autonomous AI agents")
print("Ethical Considerations Snapshot:", ethics_probe)