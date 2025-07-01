from typing import Dict, List
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from firecrawl import FirecrawlApp
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

    def generate_urls(self, job_title: str, location: str) -> List[str]:
        formatted_job_title = job_title.lower().replace(" ", "+")
        formatted_location = location.lower().replace(" ", "-")

        urls = []

        european_countries = [
            "germany", "france", "italy", "norway", "netherlands", "sweden", "denmark", "finland", "iceland", "switzerland", "austria", "belgium", "ireland", "spain", "portugal"
        ]
        scandinavian_countries = ["norway", "sweden", "denmark", "finland", "iceland"]

        # Determine which countries to use
        if location.lower() == "europe":
            countries = european_countries
        elif location.lower() == "scandinavia":
            countries = scandinavian_countries
        else:
            countries = [location.lower()]

        # Build up to 10 URLs
        for country in countries:
            url = f"https://eurojobs.com/search-results-jobs/?action=search&listing_type%5Bequal%5D=Job&keywords%5Ball_words%5D={formatted_job_title}&Location%5Blocation%5D%5Bvalue%5D={country}&Location%5Blocation%5D%5Bradius%5D=25"
            urls.append(url)
            if len(urls) >= 10:
                break  # Firecrawl beta limit

        return urls

    def find_jobs(self, job_title: str, location: str, experience_years: int, skills: List[str]) -> Dict:
        formatted_job_title = job_title.lower().replace(" ", "-")
        formatted_location = location.lower().replace(" ", "-")
        skills_string = ", ".join(skills)

        urls = self.generate_urls(job_title, location)

        if not urls:
            return {
                "result": {
                    "message": f"No URLs could be constructed for the location: {location}",
                    "raw": {},
                    "status": "error"
                }
            }

        try:
            raw_response = self.firecrawl.extract(
                urls=urls,
                prompt=f"""
Extract job postings related to "{job_title}" roles in or near "{location}". Include remote roles if relevant. 
Return a list with:
- region
- role
- job_title
- experience
- job_link

Max 10 items. Skip duplicates. Keep results relevant to user's experience and skills.
""",
                schema=ExtractSchema.model_json_schema()
            )

            json_data = raw_response.model_dump() if hasattr(raw_response, "model_dump") else json.loads(raw_response)
            job_data = json_data.get("data", {}).get("job_postings", [])

            if not job_data:
                return {
                    "result": {
                        "message": "❌ No job postings found. Try changing the job title or country.",
                        "raw": json_data,
                        "status": "no_data"
                    }
                }

            filtering_instructions = f"""
{job_data}

As a career expert, analyze these jobs:

1. SELECTED JOB OPPORTUNITIES
• Prioritize jobs with:
  - Title similar to '{job_title}'
  - Location near '{location}'
  - Experience around {experience_years} years
  - Skills: {skills_string}
• If experience or skills are missing, still include jobs if title and location match.
• Select 15–20 best matches.

2. SKILLS MATCH ANALYSIS
3. RECOMMENDATIONS
4. APPLICATION TIPS
"""

            analysis = self.agent.run(filtering_instructions)

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
