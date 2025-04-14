from flask import Flask, request, jsonify, make_response
from job_hunt_agent_new import JobHuntingAgent
import os

app = Flask(__name__)

# Load API keys from environment or hardcode for testing
MODEL_ID = os.getenv("OPENAI_MODEL_ID", "o3-mini")

agent = JobHuntingAgent(
    firecrawl_api_key=FIRECRAWL_API_KEY,
    openai_api_key=OPENAI_API_KEY,
    model_id=MODEL_ID
)

# Helper to add CORS headers
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:10013"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.after_request
def after_request_func(response):
    return add_cors_headers(response)

@app.route("/api/find-jobs", methods=["POST", "OPTIONS"])
def find_jobs():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response())

    data = request.json
    job_title = data.get("job_title")
    location = data.get("location")
    experience_years = data.get("experience_years", 0)
    skills = data.get("skills", [])

    if not job_title or not location:
        return jsonify({"error": "Missing job_title or location"}), 400

    try:
        result = agent.find_jobs(job_title, location, experience_years, skills)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/industry-trends", methods=["POST", "OPTIONS"])
def industry_trends():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response())

    data = request.json
    category = data.get("category")

    if not category:
        return jsonify({"error": "Missing category"}), 400

    try:
        trends = agent.get_industry_trends(category)
        return jsonify({"result": trends})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render will set PORT
    app.run(host='0.0.0.0', port=port)
