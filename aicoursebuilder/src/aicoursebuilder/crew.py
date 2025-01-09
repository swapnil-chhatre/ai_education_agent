from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from crewai_tools import YoutubeVideoSearchTool, WebsiteSearchTool, SerperDevTool
from crewai.tools import BaseTool
import requests
import os




class MediumSearchTool(BaseTool):
    name: str = "MediumSearchTool"
    description: str = "Searches for the top most relevant Medium articles based on a keyword search query."

    def _run(self, keyword: str) -> str:
        try:
            url = "https://medium2.p.rapidapi.com/search"
            headers = {
                "x-rapidapi-host": "medium2.p.rapidapi.com",
                "x-rapidapi-key": os.getenv("RAPID_API_KEY")
            }
            params = {"q": keyword}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            articles = response.json().get('articles', [])[:2]
            if not articles:
                return "No articles found for the given keyword."
                
            return "\n".join([f"Title: {article['title']}\nLink: {article['link']}" for article in articles])
            
        except requests.exceptions.RequestException as e:
            return f"Error making request: {str(e)}"
        except (KeyError, ValueError) as e:
            return f"Error parsing response: {str(e)}"



# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


# Creating tools 

YouTube = YoutubeVideoSearchTool()
URLSearch = SerperDevTool(
	n_results=3 # 3 URLs per search request -> 1 request per subtopic 
)


WebsiteRAG = WebsiteSearchTool()

# Medium tool 

# Google tool: Objective is to search Google for articles related to the subtopics learning objectives 
# How can it do this? - By looking up URLs and summarising the content of the articles OR by using Perplexity to search for the best articles 

""""
Can: 
1. SerperDevTool - This tool is designed to perform a semantic search for a specified query from a text’s content across the internet.
It utilizes the serper.dev API to fetch and display the most relevant search results based on the query provided by the user.

2. Perplexity - Find the best articles for subtopic XYZ 
"""


Medium = MediumSearchTool()




@CrewBase
class Aicoursebuilder():
	"""Aicoursebuilder crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
 
	# Agent #1: Planner - Responsible for planning the course
	@agent
	def planner(self) -> Agent:
		return Agent(
			config=self.agents_config['planner'],
			verbose=True,
			# llm= 'gpt-o1-mini' # Using the o1-mini model for better reasoning 
		)

	# Agent #2: Researcher - Responsible for researching for the course material
	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			verbose=True,
			tools=[YouTube, URLSearch, Medium]
		)

	# Agent #3: Organiser - Responsible for choosing the best course material 
	@agent
	def organiser(self) -> Agent:
		return Agent(
			config=self.agents_config['organiser'],
			verbose=True
		)

		#TOOLS - WebsiteSearchTool - Does RAG in the website
  
  
	# Agent #4: Checker - Responsible for checking to see if the course material satisfies the planner's requirements
	@agent
	def checker(self) -> Agent:
		return Agent(
			config=self.agents_config['checker'],
			verbose=True
		)
  
	# Agent #5: Writer - Responsible for writing the course material into a markdown file in desired format 
	@agent
	def writer(self) -> Agent:
		return Agent(
			config=self.agents_config['writer'],
			verbose=True
		)	
  
	# Agent #6: Reviewer - Responsible for reviewing the course material and critiquing it
	@agent
	def reviewer(self) -> Agent:
		return Agent(
			config=self.agents_config['reviewer'],
			verbose=True
		)



	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
 
	# Task #1: Planning Task - Breaks down the topic into a list of subtopics with dotpoints 
	@task
	def planning_task(self) -> Task:
		return Task(
			config=self.tasks_config['planning_task'],
		)
  
	# Task #2: Research Task - Researches the subtopics and creates a list of resources to be used in the course
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
		)
  
	# Task #3: Organising Task - Chooses the best resources to be used in the course
	@task
	def organising_task(self) -> Task:
		return Task(
			config=self.tasks_config['organising_task'],
		)
  
	# Task #4: Checking Task - Checks to see if the resources satisfy the planner's requirements
	@task
	def checking_task(self) -> Task:
		return Task(
			config=self.tasks_config['checking_task'],
		)
  
	# Task #5: Writing Task - Writes the course material into a markdown file in desired format
	@task
	def writing_task(self) -> Task:
		return Task(
			config=self.tasks_config['writing_task'],
		)
  
	# Task #6: Reviewing Task - Reviews the course material and critiques it
	@task
	def reviewing_task(self) -> Task:
		return Task(
			config=self.tasks_config['reviewing_task'],
		)
  
  
  

	@crew
	def crew(self) -> Crew:
		"""Creates the Aicoursebuilder crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
