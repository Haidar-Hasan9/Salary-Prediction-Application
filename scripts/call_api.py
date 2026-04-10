import requests
import time
from typing import List, Dict

# Your local API endpoint (change to deployed URL later)
API_BASE_URL = "http://localhost:8000"

# Define representative input combinations (covers the input space)
experience_levels = ["EN", "MI", "SE", "EX"]
job_titles = ["Data Scientist", "Machine Learning Engineer", "Data Analyst", "Data Engineer"]
work_years = [2020, 2021, 2022, 2023, 2024]
employee_residences = ["US", "GB", "DE", "IN", "BR"]  # mix of high and low cost

def call_predict_api(experience_level, job_title, work_year, employee_residence):
    """Send GET request to /predict endpoint and return response or error."""
    endpoint = f"{API_BASE_URL}/predict"
    params = {
        "experience_level": experience_level,
        "job_title": job_title,
        "work_year": work_year,
        "employee_residence": employee_residence
    }
    
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Connection error – is the API running?"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    print("=" * 60)
    print("Calling Salary Prediction API – covering input space")
    print("=" * 60)
    
    total_calls = 0
    successful_calls = 0
    failed_calls = 0
    
    # Loop through all combinations (not exhaustive but representative)
    for exp in experience_levels:
        for title in job_titles:
            for year in work_years:
                for country in employee_residences:
                    total_calls += 1
                    result = call_predict_api(exp, title, year, country)
                    
                    if result["success"]:
                        successful_calls += 1
                        pred = result["data"]["predicted_salary_usd"]
                        print(f"✅ {exp} | {title} | {year} | {country} → ${pred:,.2f}")
                    else:
                        failed_calls += 1
                        print(f"❌ {exp} | {title} | {year} | {country} → ERROR: {result['error']}")
                    
                    # Small delay to avoid overwhelming local server
                    time.sleep(0.1)
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {total_calls} total calls, {successful_calls} succeeded, {failed_calls} failed")
    print("=" * 60)

if __name__ == "__main__":
    main()