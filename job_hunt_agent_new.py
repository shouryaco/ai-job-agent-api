from typing import Dict, List
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from firecrawl import FirecrawlApp
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
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

        urls = [
            f"https://www.linkedin.com/jobs/search/?keywords={formatted_job_title}&location={formatted_location}&geoId=101452733",
            f"https://www.indeed.com/jobs?q={formatted_job_title}&l={formatted_location}",
            f"https://www.eurojobs.com/job-search/{formatted_job_title}/{formatted_location}/",
            f"https://www.stepstone.de/en/job-search/{formatted_job_title}/{formatted_location}/",
            f"https://www.monster.lu/en/jobs/search/?q={formatted_job_title}&where={formatted_location}",
            f"https://www.jobserve.com/gb/en/Job-Search/",
            f"https://jobs.euractiv.com/search?keywords={formatted_job_title}&location={formatted_location}",
            f"https://ec.europa.eu/eures/portal/jv-se/home",
        ]

        try:
            raw_response = self.firecrawl.extract(
                urls=urls,
                params={
                    'prompt': f"""Extract job postings ONLY from European countries.

Filter jobs based on:
- Job Title: Related to {job_title}
- Location: Europe or marked as Remote
- Experience: Around {experience_years} years
- Skills: Some of {skills_string}
- Job Type: Full-time, Part-time, Contract, Temporary, Internship

Return MAX 10 jobs with:
- region, role, job_title, experience, job_link
""",
                    'schema': ExtractSchema.model_json_schema()
                }
            )

            if isinstance(raw_response, dict) and raw_response.get('success'):
                jobs = raw_response['data'].get('job_postings', [])
            else:
                jobs = []

            if not jobs:
                return "No European job listings found matching your criteria. Try different search parameters."

            analysis = self.agent.run(
                f"""Analyze these European jobs:

{jobs}

Return:

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
                params={
                    'prompt': f"""Extract industry trends for {job_category}:
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
                    f"""Analyze these trends for {job_category}:

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

def main():
    st.set_page_config(page_title="AI Job Hunting Assistant", page_icon="💼", layout="wide")
    load_dotenv()

    env_firecrawl_key = os.getenv("FIRECRAWL_API_KEY", "")
    env_openai_key = os.getenv("OPENAI_API_KEY", "")
    default_model = os.getenv("OPENAI_MODEL_ID", "o3-mini")

    with st.sidebar:
        st.title("🔑 API Configuration")
        st.subheader("🤖 Model Selection")
        model_id = st.selectbox("Choose OpenAI Model", ["o3-mini", "gpt-4o-mini"], index=0)
        st.session_state.model_id = model_id

        st.subheader("🔐 API Keys")
        firecrawl_key = st.text_input("Firecrawl API Key", type="password", value="" if env_firecrawl_key else "")
        openai_key = st.text_input("OpenAI API Key", type="password", value="" if env_openai_key else "")

        firecrawl_key = firecrawl_key or env_firecrawl_key
        openai_key = openai_key or env_openai_key

        if firecrawl_key and openai_key:
            st.session_state.firecrawl_key = firecrawl_key
            st.session_state.openai_key = openai_key
            create_job_agent()
        else:
            st.warning("⚠️ Missing API keys")

    st.title("💼 AI Job Hunting Assistant")
    st.info("Enter your job search criteria below to get job recommendations and industry insights.")

    col1, col2 = st.columns(2)
    with col1:
        job_title = st.text_input("Job Title", placeholder="e.g., Software Engineer")
        location = st.text_input("Location", placeholder="e.g., Germany, Remote")
    with col2:
        experience_years = st.number_input("Experience (in years)", min_value=0, max_value=30, value=2)
        skills_input = st.text_area("Skills (comma separated)", placeholder="e.g., Python, SQL")
        skills = [s.strip() for s in skills_input.split(",")] if skills_input else []

    job_category = st.selectbox("Industry/Job Category", [
        "Information Technology", "Software Development", "Data Science", "Marketing",
        "Finance", "Healthcare", "Education", "Engineering", "Sales", "Human Resources"
    ])

    if st.button("🔍 Start Job Search", use_container_width=True):
        if 'job_agent' not in st.session_state:
            st.error("⚠️ Please enter your API keys in the sidebar first!")
            return
        if not job_title or not location:
            st.error("⚠️ Please enter both job title and location!")
            return

        with st.spinner("🔍 Searching for jobs..."):
            result = st.session_state.job_agent.find_jobs(job_title, location, experience_years, skills)
            st.success("✅ Job search complete!")
            st.markdown(result)

            with st.spinner("📊 Analyzing industry trends..."):
                trends = st.session_state.job_agent.get_industry_trends(job_category)
                st.success("✅ Industry analysis complete!")
                with st.expander("📈 Industry Trends"):
                    st.markdown(trends)

if __name__ == "__main__":
    main()
