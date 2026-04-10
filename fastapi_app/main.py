from fastapi import FastAPI, HTTPException
from fastapi_app.schemas import PredictionRequest, PredictionResponse
from fastapi_app.model_loader import load_model
from fastapi_app.utils import preprocess_input

app = FastAPI(title="Salary Prediction API", description="Predict data science job salaries using a Decision Tree model")

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/predict", response_model=PredictionResponse)
async def predict(
    experience_level: str,
    job_title: str,
    work_year: int,
    employee_residence: str
):
    # Validate inputs manually (or use Pydantic query parameters; but easier to reuse schema)
    try:
        req = PredictionRequest(
            experience_level=experience_level,
            job_title=job_title,
            work_year=work_year,
            employee_residence=employee_residence
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    # Preprocess
    X = preprocess_input(req.experience_level, req.job_title, req.work_year, req.employee_residence)
    
    # Predict
    model, _ = load_model()
    pred = model.predict(X)[0]
    
    return PredictionResponse(
        experience_level=req.experience_level,
        job_title=req.job_title,
        work_year=req.work_year,
        employee_residence=req.employee_residence,
        predicted_salary_usd=round(pred, 2)
    )