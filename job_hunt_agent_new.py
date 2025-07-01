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

    def find_jobs(self, job_title: str, location: str, experience_years: int, skills: List[str]) -> Dict:
        formatted_job_title = job_title.lower().replace(" ", "-")
        formatted_location = location.lower().replace(" ", "-")
        skills_string = ", ".join(skills)

        urls = []

        if location.lower() in ["germany", "france", "italy", "norway", "europe", "netherlands"]:
            urls.append(f"https://eurojobs.com/search-results-jobs/?action=search&listing_type%5Bequal%5D=Job&keywords%5Ball_words%5D=any&Location%5Blocation%5D%5Bvalue%5D={location}&Location%5Blocation%5D%5Bradius%5D=10")
        if location.lower() in ["usa", "united states"]:
            urls.append(f"https://www.indeed.com/jobs?q={formatted_job_title}&l={formatted_location}")
        if location.lower() in ["india"]:
            urls.append(f"https://www.naukri.com/{formatted_job_title}-jobs-in-{formatted_location}")

        try:
            raw_response = self.firecrawl.extract(
                urls=urls,
                prompt=f"""
Extract job postings related to {job_title} roles in {location}. Include remote roles. 
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
                        "message": "Firecrawl could not extract valid job listings. Try adjusting the prompt, job title, or check API access to the job boards.",
                        "raw": json_data,
                        "status": "no_data"
                    }
                }

            # Always include jobs that partially match, but prioritize strict matches if job_title is not 'any'
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
                    "message": f"An error occurred while searching for jobs: {str(e)}",
                    "raw": {},
                    "status": "error"
                }
            }
