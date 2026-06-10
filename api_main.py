from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import create_engine, text
from datetime import date, timedelta
from typing import Optional, List
from pydantic import BaseModel
import uvicorn

# --- Database Setup ---
DB_URL = "postgresql://postgres:finops_password@localhost:5432/finops_intelligence"
engine = create_engine(DB_URL, pool_size=5, max_overflow=10)

app = FastAPI(
    title="Cloud FinOps Intelligence API",
    description="Middleware for serving ML-scored cloud billing data and forecasts.",
    version="1.0.0"
)

# --- Pydantic Response Models ---
class SpendRecord(BaseModel):
    timestamp: str
    cloud_provider: str
    unified_category: str
    cost: float
    is_anomaly: int
    anomaly_type: str

class ForecastRecord(BaseModel):
    date: str
    predicted_cost: float
    lower_bound: float
    upper_bound: float
    model_used: str

# --- Endpoints ---

@app.get("/api/v1/health")
def health_check():
    """Verifies API and Database connectivity."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")

@app.get("/api/v1/spend", response_model=List[SpendRecord])
def get_actual_spend(
    provider: str = Query(..., description="E.g., AWS, GCP, Azure"),
    category: str = Query(..., description="E.g., COMPUTE, STORAGE, NETWORK"),
    start_date: date = Query(..., description="Start of the query window"),
    end_date: date = Query(..., description="End of the query window")
):
    """Fetches normalized historical spend and anomaly flags."""
    query = text("""
        SELECT timestamp, cloud_provider, unified_category, cost, is_anomaly, anomaly_type
        FROM cloud_spend
        WHERE cloud_provider = :provider 
          AND unified_category = :category
          AND timestamp >= :start_date 
          AND timestamp <= :end_date
        ORDER BY timestamp ASC;
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {
            "provider": provider, 
            "category": category, 
            "start_date": start_date, 
            "end_date": end_date
        }).mappings().all()
        
    return [{"timestamp": str(r["timestamp"]), **r} for r in result]

# Note: In a full production setup, we would load the forecast from a new DB hypertable. 
# For this immediate module, we will mock the connection to the CSV we just generated.
import pandas as pd
@app.get("/api/v1/forecast", response_model=List[ForecastRecord])
def get_forecast():
    """Fetches the 90-day ensembled prediction boundary."""
    try:
        df = pd.read_csv("production_forecast.csv")
        return df.to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Forecast data not yet generated.")

if __name__ == "__main__":
    print("--- Starting FinOps API Middleware ---")
    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=True)