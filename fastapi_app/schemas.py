from pydantic import BaseModel, Field, validator
from typing import Optional

class PredictionRequest(BaseModel):
    experience_level: str = Field(..., description="EN, MI, SE, or EX")
    job_title: str = Field(..., description="e.g., Data Scientist")
    work_year: int = Field(..., ge=2020, le=2025)
    employee_residence: str = Field(..., min_length=2, max_length=2, description="ISO country code like US, GB, DE")
    
    @validator('experience_level')
    def validate_experience_level(cls, v):
        allowed = {'EN', 'MI', 'SE', 'EX'}
        if v not in allowed:
            raise ValueError(f'experience_level must be one of {allowed}')
        return v
    
    # You can add more validations for job_title (optional) – but many titles possible, so skip strict validation.

class PredictionResponse(BaseModel):
    experience_level: str
    job_title: str
    work_year: int
    employee_residence: str
    predicted_salary_usd: float