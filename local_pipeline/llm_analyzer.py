import pandas as pd
import matplotlib.pyplot as plt
import ollama

import base64

import os
import subprocess
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# Ollama config
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Supabase config (must be in .env)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ Supabase credentials missing. Skipping database storage.")
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def generate_predictions_df():
    """Simulate predictions for a representative set of job profiles."""
    data = [
        {"experience_level": "EN", "job_title": "Data Scientist", "work_year": 2023, "employee_residence": "US", "predicted_salary_usd": 72442},
        {"experience_level": "MI", "job_title": "Data Scientist", "work_year": 2023, "employee_residence": "US", "predicted_salary_usd": 96360},
        {"experience_level": "SE", "job_title": "Data Scientist", "work_year": 2023, "employee_residence": "US", "predicted_salary_usd": 139775},
        {"experience_level": "EX", "job_title": "Data Scientist", "work_year": 2023, "employee_residence": "US", "predicted_salary_usd": 176200},
        {"experience_level": "SE", "job_title": "Machine Learning Engineer", "work_year": 2023, "employee_residence": "US", "predicted_salary_usd": 139775},
        {"experience_level": "SE", "job_title": "Data Analyst", "work_year": 2023, "employee_residence": "US", "predicted_salary_usd": 139775},
        {"experience_level": "SE", "job_title": "Data Scientist", "work_year": 2023, "employee_residence": "IN", "predicted_salary_usd": 59773},
    ]
    return pd.DataFrame(data)

def generate_llm_analysis(df: pd.DataFrame):
    """Send data to Ollama and get narrative + chart code."""
    summary = df.groupby("experience_level")["predicted_salary_usd"].mean().round(2).to_dict()
    summary_text = ", ".join([f"{k}: ${v:,.0f}" for k, v in summary.items()])
    
    prompt = f"""
You are a data analyst. Given the following average salaries by experience level for data science jobs (in USD):
{summary_text}

Write a short narrative (2-3 paragraphs) that:
- Compares salaries across experience levels.
- Explains why senior roles earn more.
- Mentions how location (US vs IN) affects salary.
- Gives actionable advice for someone looking to increase their salary.

Then, generate Python matplotlib code (inside triple backticks) that creates a bar chart of average salary by experience level.
The code should be self-contained and use the data provided: 
experience_levels = ['EN','MI','SE','EX']
avg_salaries = [72442, 96360, 139775, 176200]

**Important:** The code must save the chart to a file named 'chart.png' using plt.savefig('chart.png'), and should NOT call plt.show().

Example:
import matplotlib.pyplot as plt
experience_levels = ['EN','MI','SE','EX']
avg_salaries = [72442, 96360, 139775, 176200]
plt.figure(figsize=(8,5))
plt.bar(experience_levels, avg_salaries)
plt.xlabel('Experience Level')
plt.ylabel('Average Salary (USD)')
plt.title('Average Salary by Experience Level')
plt.savefig('chart.png')
"""
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': prompt}])
    llm_output = response['message']['content']
    
    narrative = llm_output.split("```python")[0].strip()
    
    chart_code = ""
    if "```python" in llm_output:
        code_part = llm_output.split("```python")[1].split("```")[0]
        chart_code = code_part
    else:
        chart_code = """
import matplotlib.pyplot as plt
experience_levels = ['EN','MI','SE','EX']
avg_salaries = [72442, 96360, 139775, 176200]
plt.figure(figsize=(8,5))
plt.bar(experience_levels, avg_salaries, color='skyblue')
plt.xlabel('Experience Level')
plt.ylabel('Average Salary (USD)')
plt.title('Average Salary by Experience Level')
plt.savefig('chart.png')
"""
    return narrative, chart_code

def execute_chart_code(chart_code: str, output_image="chart.png"):
    """Execute the chart code and return base64 encoded image."""
    
    with open("temp_chart.py", "w") as f:
        f.write(chart_code)
    

    result = subprocess.run(["python", "temp_chart.py"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error executing chart code:")
        print(result.stderr)
        # Fallback
        experience_levels = ['EN', 'MI', 'SE', 'EX']
        avg_salaries = [72442, 96360, 139775, 176200]
        plt.figure(figsize=(8,5))
        plt.bar(experience_levels, avg_salaries, color='skyblue')
        plt.xlabel('Experience Level')
        plt.ylabel('Average Salary (USD)')
        plt.title('Average Salary by Experience Level')
        plt.savefig(output_image)
    

    os.remove("temp_chart.py")
    

    with open(output_image, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    return img_base64

def store_prediction(exp_level, job_title, work_year, residence, salary, narrative, chart_b64):
    """Insert a single prediction into Supabase."""
    if supabase is None:
        print("⚠️ Supabase not configured – skipping storage")
        return
    data = {
        "experience_level": exp_level,
        "job_title": job_title,
        "work_year": work_year,
        "employee_residence": residence,
        "predicted_salary_usd": salary,
        "llm_narrative": narrative,
        "chart_base64": chart_b64
    }
    try:
        supabase.table("predictions").insert(data).execute()
        print(f"✅ Stored: {exp_level} | {job_title} | {residence} | ${salary:,.2f}")
    except Exception as e:
        print(f"❌ Store failed for {job_title}: {e}")

if __name__ == "__main__":
    # 1. Generate predictions DataFrame
    df = generate_predictions_df()
    print("Data for analysis:")
    print(df)
    
    # 2. Get LLM narrative and chart code
    narrative, chart_code = generate_llm_analysis(df)
    print("\n" + "="*60)
    print("LLM NARRATIVE:")
    print("="*60)
    print(narrative)
    
    print("\n" + "="*60)
    print("GENERATED CHART CODE:")
    print("="*60)
    print(chart_code)
    
    # 3. Generate chart and get base64
    img_base64 = execute_chart_code(chart_code)
    print("\n✅ Chart generated and encoded to base64.")
    
    # 4. Store each row from df into Supabase (all share same narrative and chart)
    for _, row in df.iterrows():
        store_prediction(
            exp_level=row['experience_level'],
            job_title=row['job_title'],
            work_year=row['work_year'],
            residence=row['employee_residence'],
            salary=row['predicted_salary_usd'],
            narrative=narrative,
            chart_b64=img_base64
        )
    
    print("\n🎉 All predictions stored in Supabase. Ready for dashboard!")