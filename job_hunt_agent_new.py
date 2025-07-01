from typing import Dict, List
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from firecrawl import FirecrawlApp
import json
import urllib.parse

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
        formatted_job_title = urllib.parse.quote_plus(job_title.lower().strip())
        formatted_location = urllib.parse.quote_plus(location.lower().strip())
        skills_string = ", ".join(skills)

        urls = []

        # EuroJobs: For Europe
        if location.lower() in ["germany", "france", "italy", "norway", "europe", "netherlands", "spain"]:
            urls.append(
                f"https://eurojobs.com/search-results-jobs/?action=search&listing_type%5Bequal%5D=Job"
                f"&keywords%5Ball_words%5D={formatted_job_title}"
                f"&Location%5Blocation%5D%5Bvalue%5D={formatted_location}"
                f"&Location%5Blocation%5D%5Bradius%5D=25"
            )

        # Indeed: USA
        if location.lower() in ["usa", "united states"]:
            urls.append(f"https://www.indeed.com/jobs?q={formatted_job_title}&l={formatted_location}")

        # Naukri: India
        if location.lower() in ["india"]:
            urls.append(f"https://www.naukri.com/{formatted_job_title}-jobs-in-{formatted_location}")

        if not urls:
            return {
                "result": {
                    "message": "⚠️ Could not determine job source for this location.",
                    "raw": {},
                    "status": "error"
                }
            }

        try:
            # 🔥 Firecrawl extract
            prompt = f"""
Extract job postings for '{job_title}' roles in '{location}' with preference for remote options.
Return a list of:
- region
- role
- job_title
- experience
- job_link
Only extract up to 15–20 relevant job entries.
"""

            raw_response = self.firecrawl.extract(
                urls=urls,
                prompt=prompt.strip(),
                schema=ExtractSchema.model_json_schema()
            )

            # ✅ Pydantic-based serialization
            json_data = raw_response.model_dump() if hasattr(raw_response, "model_dump") else json.loads(raw_response)
            job_data = json_data.get("data", {}).get("job_postings", [])

            if not job_data:
                return {
                    "result": {
                        "message": "❌ Firecrawl could not extract valid job listings. Try adjusting the job title, location, or experience.",
                        "raw": json_data,
                        "status": "no_data"
                    }
                }

            # 🎯 AI-based Filtering & Suggestions
            filtering_instructions = f"""
{job_data}

Analyze and select jobs that match:
- Title similar to: "{job_title}"
- Located in or near: "{location}"
- Experience: around {experience_years} year(s)
- Skills: {skills_string if skills else 'No skills provided'}

Steps:
1. SELECTED JOB OPPORTUNITIES (Top 15–20 matches)
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
