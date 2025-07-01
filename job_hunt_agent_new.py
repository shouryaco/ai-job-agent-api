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

        # More specific EuroJobs search with keywords included
        if location.lower() in ["germany", "france", "italy", "norway", "europe", "netherlands"]:
            urls.append(
                f"https://eurojobs.com/search-results-jobs/?action=search"
                f"&listing_type%5Bequal%5D=Job"
                f"&keywords%5Ball_words%5D={job_title.replace(' ', '+')}"
                f"&Location%5Blocation%5D%5Bvalue%5D={location}"
                f"&Location%5Blocation%5D%5Bradius%5D=25"
            )
        if location.lower() in ["usa", "united states"]:
            urls.append(f"https://www.indeed.com/jobs?q={formatted_job_title}&l={formatted_location}")
        if location.lower() in ["india"]:
            urls.append(f"https://www.naukri.com/{formatted_job_title}-jobs-in-{formatted_location}")

        try:
            # Firecrawl Extraction
            raw_response = self.firecrawl.extract(
                urls=urls,
                prompt=f"""
You are a job extraction agent. Extract job listings from the page and return a list of up to 10 job postings.

Each job should include:
- job_title
- role
- region
- experience (if visible)
- job_link

Filter for jobs related to: "{job_title}" in "{location}" with around {experience_years} years of experience.
Include remote jobs. If experience or skills are not listed, still include the job.

Skip duplicates. Return in structured JSON format.
""",
                schema=ExtractSchema.model_json_schema()
            )

            # Convert to JSON-safe dict
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

            # Relaxed filtering: no skills/title strict matching — just analysis
            filtering_instructions = f"""
{job_data}

You are a career job analyst.

1. **Selected Job Opportunities**
- Show 10–15 jobs related to "{job_title}" in "{location}"
- Include jobs with or without experience or skills
- Prioritize job title or role similarity if present

2. **Summary**
3. **Recommendations**
4. **Tips**
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
