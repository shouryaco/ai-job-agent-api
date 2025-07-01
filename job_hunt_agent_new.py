from typing import Dict, List
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from firecrawl import FirecrawlApp
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class NestedModel1(BaseModel):
    region: str = Field(default=None)
    role: str = Field(default=None)
    job_title: str = Field(default=None)
    experience: str = Field(default=None)
    job_link: str = Field(default=None)

class ExtractSchema(BaseModel):
    job_postings: List[NestedModel1]

class IndustryTrend(BaseModel):
    industry: str = Field(default=None)
    avg_salary: float = Field(default=None)
    growth_rate: float = Field(default=None)
    demand_level: str = Field(default=None)
    top_skills: List[str] = Field(default=None)

class IndustryTrendsSchema(BaseModel):
    industry_trends: List[IndustryTrend]

class JobHuntingAgent:
    def __init__(self, firecrawl_api_key: str, openai_api_key: str, model_id: str = "o3-mini"):
        self.agent = Agent(
            model=OpenAIChat(id=model_id, api_key=openai_api_key),
            markdown=True,
            description="Career expert helping users find job opportunities."
        )
        self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)

    def find_jobs(self, job_title: str, location: str, experience_years: int, skills: List[str]) -> str:
        formatted_job_title = job_title.lower().replace(" ", "-")
        formatted_location = location.lower().replace(" ", "-")
        skills_string = ", ".join(skills)

        urls = [
            f"https://www.naukri.com/{formatted_job_title}-jobs-in-{formatted_location}",
            f"https://www.indeed.com/jobs?q={formatted_job_title}&l={formatted_location}",
            f"https://www.monster.com/jobs/search/?q={formatted_job_title}&where={formatted_location}",
            f"https://eurojobs.com/search-results-jobs/?action=search&listing_type%5Bequal%5D=Job&keywords%5Ball_words%5D={formatted_job_title or 'any'}&Location%5Blocation%5D%5Bvalue%5D={location}"
        ]

        try:
            raw_response = self.firecrawl.extract(
                urls=urls,
                prompt=f"""
                Extract up to 10 job postings from these job boards, prioritizing matches for:

                - Job Title: Related to '{job_title}'
                - Location: {location} (Europe-focused)
                - Skills: At least one of: {skills_string}
                - Experience: Around {experience_years} years (optional)

                For each job, extract as much of this as available:
                - region
                - role
                - job_title
                - experience
                - job_link

                If experience or skills are not listed, leave them blank — do not skip those jobs.
                """,
                schema=ExtractSchema.model_json_schema()
            )

            if isinstance(raw_response, dict):
                jobs = raw_response.get("data", {}).get("job_postings", [])
            else:
                jobs = raw_response.data.get("job_postings", [])

            if not jobs:
                return "No job listings found matching your criteria. Try adjusting your search parameters."

            strict_mode = job_title.lower() != "any" and skills and experience_years > 0

            filtering_instructions = f"""
1. STRICTLY analyze jobs that satisfy:
   - Job Title: Must closely match '{job_title}'
   - Location: Must be in or near '{location}'
   - Experience: Must be around {experience_years} years
   - Skills: Must include one or more of: {skills_string}
2. DO NOT include jobs missing experience or skills
3. Select 5-6 jobs that best match
            """ if strict_mode else f"""
1. Analyze jobs that roughly match:
   - Job Title: Similar to '{job_title}'
   - Location: Near '{location}'
   - Experience: Preferably around {experience_years} years
   - Skills: Preferably one of: {skills_string}
2. Include incomplete jobs if others are missing
3. Select up to 6 jobs that best match
            """

            analysis_prompt = f"""
As a career expert, analyze these job opportunities:

Jobs Found (JSON):
{jobs}

{filtering_instructions}

💼 SELECTED JOB OPPORTUNITIES
- Job Title, Role, Region, Experience, Link

🔍 SKILLS MATCH ANALYSIS
- Match with user skills and experience

💡 RECOMMENDATIONS
- Top 3 jobs with reason

📝 APPLICATION TIPS
- Tips per job type
            """

            analysis = self.agent.run(analysis_prompt)
            return analysis.content

        except Exception as e:
            return f"An error occurred while searching for jobs: {str(e)}"

    def get_industry_trends(self, job_category: str) -> str:
        urls = [
            f"https://www.payscale.com/research/US/Job={job_category.replace(' ', '_')}/Salary",
            f"https://www.glassdoor.com/Salaries/{job_category.lower().replace(' ', '-')}-salary-SRCH_KO0,{len(job_category)}.htm"
        ]

        try:
            raw_response = self.firecrawl.extract(
                urls=urls,
                prompt=f"""
                Extract industry trends for the category '{job_category}'.
                For each trend, provide:
                - industry
                - avg_salary
                - growth_rate
                - demand_level
                - top_skills

                Provide 3-5 insights across sub-domains if available.
                """,
                schema=IndustryTrendsSchema.model_json_schema()
            )

            if isinstance(raw_response, dict):
                trends = raw_response.get("data", {}).get("industry_trends", [])
            else:
                trends = raw_response.data.get("industry_trends", [])

            if not trends:
                return f"No industry trends found for {job_category}."

            analysis = self.agent.run(f"""
Analyze these industry trends:

{trends}

📊 INDUSTRY TRENDS SUMMARY
- Bullet summary of salary and demand

🔥 TOP SKILLS IN DEMAND
- Bullet list

📈 CAREER GROWTH OPPORTUNITIES
- Fast-growing subfields

🎯 RECOMMENDATIONS FOR JOB SEEKERS
- Strategic tips
            """)

            return analysis.content

        except Exception as e:
            return f"An error occurred while fetching trends: {str(e)}"

def create_job_agent():
    if 'job_agent' not in st.session_state:
        st.session_state.job_agent = JobHuntingAgent(
            firecrawl_api_key=st.session_state.firecrawl_key,
            openai_api_key=st.session_state.openai_key,
            model_id=st.session_state.model_id
        )
