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

    def build_eurojobs_url(self, country: str) -> str:
        return (
            "https://eurojobs.com/search-results-jobs/"
            "?action=search"
            "&listing_type%5Bequal%5D=Job"
            "&keywords%5Ball_words%5D=any"
            f"&Location%5Blocation%5D%5Bvalue%5D={country}"
            "&Location%5Blocation%5D%5Bradius%5D=10"
        )

    def find_jobs(self, job_title: str, location: str, experience_years: int, skills: List[str]) -> str:
        formatted_job_title = job_title.lower().replace(" ", "-")
        formatted_location = location.lower().replace(" ", "-")
        skills_string = ", ".join(skills)

        urls = []
        if "europe" in location.lower():
            urls = [
                self.build_eurojobs_url("Germany"),
                self.build_eurojobs_url("France"),
                self.build_eurojobs_url("Sweden")
            ]
        elif any(loc in location.lower() for loc in ["germany", "france", "norway", "sweden", "netherlands", "italy", "spain"]):
            urls = [self.build_eurojobs_url(location.title())]
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
                prompt=f"""
From the pages below, extract up to 10 job listings.

Try to extract (if available):
- job_title: the title of the job
- role: the functional role (e.g., 'Frontend Developer')
- region: city, country, or general location
- experience: required experience (e.g., '3+ years')
- job_link: the direct URL to apply

You can leave missing fields blank.

Only extract real jobs, skip ads or navigation.

User is looking for:
• Title: {job_title}
• Location: {location}
• Experience: ~{experience_years} years
• Skills: {skills_string}
""",
                schema=ExtractSchema.model_json_schema()
            )

            if isinstance(raw_response, dict) and raw_response.get('success'):
                jobs = raw_response['data'].get('job_postings', [])
            else:
                jobs = []

            if not jobs:
                return {
                    "status": "no_data",
                    "message": "Firecrawl could not extract valid job listings. Try adjusting the prompt, job title, or check API access to the job boards.",
                    "raw": raw_response.dict() if hasattr(raw_response, 'dict') else str(raw_response)
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
                prompt=f"""
Extract industry trends for {job_category}:
- industry, avg_salary, growth_rate, demand_level, top_skills
Minimum 3 roles or sub-industries
""",
                schema=IndustryTrendsSchema.model_json_schema()
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
