from typing import Dict, List
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from firecrawl import FirecrawlApp
from urllib.parse import quote_plus
import json

class NestedModel1(BaseModel):
    region: str = Field(description="Region or area where the job is located", default=None)
    role: str = Field(description="Specific role or function within the job category", default=None)
    job_title: str = Field(description="Title of the job position", default=None)
    experience: str = Field(description="Experience required for the position", default=None)
    job_link: str = Field(description="Link to the job posting", default=None)

class ExtractSchema(BaseModel):
    job_postings: List[NestedModel1] = Field(description="List of job postings")

class JobHuntingAgent:
    def __init__(self, firecrawl_api_key: str, openai_api_key: str, model_id: str = "gpt-4o"):
        self.agent = Agent(
            model=OpenAIChat(id=model_id, api_key=openai_api_key),
            markdown=True,
            description="I help find and analyze job opportunities."
        )
        self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)

    def find_jobs(self, job_title: str, location: str, experience_years: int, skills: List[str]) -> Dict:
        formatted_job_title = quote_plus(job_title)
        skills_string = ", ".join(skills)
        location_lower = location.lower()

        # Countries for 'Europe'
        EUROPEAN_COUNTRIES = [
            "Germany", "France", "Italy", "Norway", "Sweden", "Netherlands", "Denmark",
            "Finland", "Austria", "Belgium", "Switzerland", "Poland", "Spain", "Portugal",
            "Czech Republic", "Hungary", "Greece", "Ireland", "Romania", "Croatia"
        ]

        urls = []

        scandinavian_countries = ["sweden", "norway", "denmark", "finland", "iceland"]
        european_countries = [
            "germany", "france", "italy", "spain", "portugal",
            "netherlands", "belgium", "sweden", "norway", "denmark"
        ]

        if location.lower() == "scandinavia":
            for country in scandinavian_countries[:10]:
                urls.append(
                    f"https://eurojobs.com/search-results-jobs/?action=search&listing_type%5Bequal%5D=Job&keywords%5Ball_words%5D={formatted_job_title}&Location%5Blocation%5D%5Bvalue%5D={country}&Location%5Blocation%5D%5Bradius%5D=10"
                )

        if location_lower == "europe":
            for country in european_countries[:10]:
                urls.append(
                    f"https://eurojobs.com/search-results-jobs/?action=search&listing_type%5Bequal%5D=Job&keywords%5Ball_words%5D={formatted_job_title}&Location%5Blocation%5D%5Bvalue%5D={country}&Location%5Blocation%5D%5Bradius%5D=10"
                )
        elif location.lower() in scandinavian_countries + european_countries:
            urls.append(
                f"https://eurojobs.com/search-results-jobs/?action=search&listing_type%5Bequal%5D=Job&keywords%5Ball_words%5D={formatted_job_title}&Location%5Blocation%5D%5Bvalue%5D={formatted_location}&Location%5Blocation%5D%5Bradius%5D=10"
            )
        elif location.lower() in ["usa", "united states"]:
            urls.append(f"https://www.indeed.com/jobs?q={formatted_job_title}&l={formatted_location}")
        elif location.lower() in ["india"]:
            urls.append(f"https://www.naukri.com/{formatted_job_title}-jobs-in-{formatted_location}")

        try:
            raw_response = self.firecrawl.extract(
                urls=urls,
                prompt=f"""
Extract job postings related to {job_title} roles in {location}. Include remote roles if available.
Return a list with:
- region
- role
- job_title
- experience
- job_link
Max 10 items per source. Skip duplicates. Match results to user's input: {experience_years} years experience, skills: {skills_string}.
""",
                schema=ExtractSchema.model_json_schema()
            )

            json_data = raw_response.model_dump() if hasattr(raw_response, "model_dump") else json.loads(raw_response)
            job_data = json_data.get("data", {}).get("job_postings", [])

            if not job_data:
                return {
                    "result": {
                        "message": "❌ No jobs extracted. Try adjusting job title or location.",
                        "raw": json_data,
                        "status": "no_data"
                    }
                }

            # Prompt GPT for analysis
            analysis_prompt = f"""
{job_data}

As a career expert, analyze these jobs:

1. SELECTED JOB OPPORTUNITIES
• Prioritize jobs with:
  - Title similar to '{job_title}'
  - Location near '{location}'
  - Experience around {experience_years} years
  - Skills: {skills_string}
• Even if some experience or skills are missing, include roles with clear title and region matches.
• Select top 15–20 matches and format output like:
  1. Job Title - Role - Region - [Job Link]

2. SKILLS MATCH ANALYSIS
3. RECOMMENDATIONS
4. APPLICATION TIPS
"""

            analysis = self.agent.run(analysis_prompt)

            return {
                "result": {
                    "message": analysis.content,
                    "raw": json_data,
                    "status": "ok"
                }
            }

        except Exception as e:
            return {
                "result": {
                    "message": f"❌ Error occurred during job search: {str(e)}",
                    "raw": {},
                    "status": "error"
                }
            }
