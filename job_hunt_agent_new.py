from typing import Dict, List
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from firecrawl import FirecrawlApp
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

class NestedModel1(BaseModel):
    region: str = Field(description="Region or area where the job is located", default=None)
    role: str = Field(description="Specific role or function within the job category", default=None)
    job_title: str = Field(description="Title of the job position", default=None)
    experience: str = Field(description="Experience required for the position", default=None)
    job_link: str = Field(description="Link to the job posting", default=None)

class ExtractSchema(BaseModel):
    job_postings: List[NestedModel1] = Field(description="List of job postings")

class IndustryTrend(BaseModel):
    industry: str = Field(description="Industry name", default=None)
    avg_salary: float = Field(description="Average salary in the industry", default=None)
    growth_rate: float = Field(description="Growth rate of the industry", default=None)
    demand_level: str = Field(description="Demand level in the industry", default=None)
    top_skills: List[str] = Field(description="Top skills in demand for this industry", default=None)

class IndustryTrendsSchema(BaseModel):
    industry_trends: List[IndustryTrend] = Field(description="List of industry trends")

class JobHuntingAgent:
    def __init__(self, firecrawl_api_key: str, openai_api_key: str, model_id: str = "o3-mini"):
        self.agent = Agent(
            model=OpenAIChat(id=model_id, api_key=openai_api_key),
            markdown=True,
            description="I am a career expert who helps find and analyze job opportunities based on user preferences."
        )
        self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)

    def find_jobs(self, job_title: str, location: str, experience_years: int, skills: List[str]) -> str:
        formatted_job_title = job_title.lower().replace(" ", "-")
        formatted_location = location.lower().replace(" ", "-")
        skills_string = ", ".join(skills)

        # Determine job board URLs based on location
        urls = []
        if any(loc in location.lower() for loc in ["germany", "france", "norway", "sweden", "netherlands", "europe", "italy", "spain"]):
            urls = [
                f"https://www.eurojobs.com/job-search/{formatted_job_title}/{formatted_location}/",
                f"https://www.stepstone.de/en/job-search/{formatted_job_title}/{formatted_location}/",
                f"https://jobs.euractiv.com/search?keywords={formatted_job_title}&location={formatted_location}"
            ]
        elif "india" in location.lower():
            urls = [
                f"https://www.naukri.com/{formatted_job_title}-jobs-in-{formatted_location}",
                f"https://www.monsterindia.com/srp/results?query={formatted_job_title}&locations={formatted_location}"
            ]
        elif "usa" in location.lower() or "united states" in location.lower():
            urls = [
                f"https://www.indeed.com/jobs?q={formatted_job_title}&l={formatted_location}",
                f"https://www.monster.com/jobs/search/?q={formatted_job_title}&where={formatted_location}"
            ]
        else:
            urls = [
                f"https://www.indeed.com/jobs?q={formatted_job_title}&l={formatted_location}",
                f"https://www.monster.com/jobs/search/?q={formatted_job_title}&where={formatted_location}"
            ]

        try:
            raw_response = self.firecrawl.extract(
                urls=urls,
                params={
                    'prompt': f"""
Extract up to 10 job listings from the pages below.
Try your best to extract:
- region (e.g., country or city)
- role (e.g., 'Software Developer')
- job_title (e.g., 'Frontend Engineer')
- experience (e.g., '2+ years', 'Entry-level')
- job_link (direct job posting URL)

The user's skills are: {skills_string}
Experience: ~{experience_years} years
""",
                    'schema': ExtractSchema.model_json_schema()
                }
            )

            print("Raw Firecrawl Data:", raw_response)

            if isinstance(raw_response, dict) and raw_response.get('success'):
                jobs = raw_response['data'].get('job_postings', [])
            else:
                jobs = []

            if not jobs:
                return {
                    "status": "no_data",
                    "message": "Firecrawl could not extract valid job listings. Try adjusting the prompt, job title, or check API access to the job boards.",
                    "raw": raw_response
                }

            analysis = self.agent.run(
                f"""
Analyze these jobs:

{jobs}

💼 SELECTED JOB OPPORTUNITIES
• Job Title & Role
• Location
• Experience
• Pros and Cons
• Job Link

🔍 SKILLS MATCH ANALYSIS
• Skills vs Requirements
• Experience Match
• Growth potential

💡 RECOMMENDATIONS
• Top 3 Jobs
• Why they stand out

📝 APPLICATION TIPS
• Resume suggestions
• Application strategy
"""
            )

            return {
                "status": "success",
                "jobs": jobs,
                "analysis": analysis.content
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def get_industry_trends(self, job_category: str) -> str:
        urls = [
            f"https://www.payscale.com/research/US/Job={job_category.replace(' ', '_')}/Salary",
            f"https://www.glassdoor.com/Salaries/{job_category.lower().replace(' ', '-')}-salary-SRCH_KO0,{len(job_category)}.htm"
        ]

        try:
            raw_response = self.firecrawl.extract(
                urls=urls,
                params={
                    'prompt': f"""
Extract industry trends for {job_category}:
- industry, avg_salary, growth_rate, demand_level, top_skills
Minimum 3 roles or sub-industries
""",
                    'schema': IndustryTrendsSchema.model_json_schema(),
                }
            )

            if isinstance(raw_response, dict) and raw_response.get('success'):
                industries = raw_response['data'].get('industry_trends', [])
                if not industries:
                    return f"No industry trends available for {job_category}."

                analysis = self.agent.run(
                    f"""
Analyze these trends for {job_category}:

{industries}

📊 INDUSTRY TRENDS SUMMARY
• Salary and demand overview

🔥 TOP SKILLS IN DEMAND
• Skills list

📈 CAREER GROWTH OPPORTUNITIES
• High growth roles
• Emerging skills

🎯 RECOMMENDATIONS FOR JOB SEEKERS
• Strategy and advice
"""
                )

                return analysis.content

            return f"No industry trends data available for {job_category}."
        except Exception as e:
            return f"An error occurred while fetching industry trends: {str(e)}"

def create_job_agent():
    if 'job_agent' not in st.session_state:
        st.session_state.job_agent = JobHuntingAgent(
            firecrawl_api_key=st.session_state.firecrawl_key,
            openai_api_key=st.session_state.openai_key,
            model_id=st.session_state.model_id
        )
