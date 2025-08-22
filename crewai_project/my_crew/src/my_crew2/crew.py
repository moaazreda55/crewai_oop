from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Keyword_output(BaseModel):
    keyword: list[str]


@CrewBase
class MyCrew():
    """MyCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    llm = LLM(model=os.environ["MODEL"], api_key=os.environ["GEMINI_API_KEY"])

    @agent
    def Keyword_Instruction(self) -> Agent:
        return Agent(
            config=self.agents_config['Keyword_Instruction'],  # type: ignore[index]
            verbose=True,
            llm=self.llm
        )
    
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],  # type: ignore[index]
            verbose=True,
            llm=self.llm
        )
    
    @agent
    def Evaluation_Agent(self) -> Agent:
        return Agent(
            config=self.agents_config['Evaluation_Agent'],  # type: ignore[index]
            verbose=True,
            llm=self.llm
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],  # type: ignore[index]
            verbose=True,
            llm=self.llm
        )

  
    @task
    def Keyword_task(self) -> Task:
        return Task(
            config=self.tasks_config['Keyword_task'],
            output_pydantic=Keyword_output
            )
         
    
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],  # type: ignore[index]
        )

    @task
    def evaluation_task(self) -> Task:
        return Task(
            config=self.tasks_config['evaluation_task'],  # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],  # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MyCrew crew"""
       

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            
        )
