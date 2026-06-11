import os
import sys
import uuid
import asyncio
from datetime import date, datetime, timedelta
from typing import List, Dict, Any

# 1. Enforce rigorous absolute module resolution paths before any business logic imports execute
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # FinSight/backend/app
PARENT_DIR = os.path.dirname(CURRENT_DIR)                 # FinSight/backend
ROOT_DIR = os.path.dirname(PARENT_DIR)                    # FinSight Root

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import pandas as pd
import jwt
from fastapi import FastAPI, HTTPException, Query, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, text
from pydantic import BaseModel, EmailStr
import uvicorn

# Clean structural imports
from core.config import settings
from core.security import hash_password, verify_password, create_access_token, decode_access_token
from api.agent_rca import app_graph

# --- Security Configuration Core ---
security_bearer = HTTPBearer()

# --- Database Connection Pool Setup ---
# Dynamically loaded from config.py -> .env
engine = create_engine(settings.DB_URL, pool_size=10, max_overflow=20)

app = FastAPI(
    title="FinSight Intelligence Core",
    description="Enterprise Multi-Tenant Cloud FinOps Middleware Gateway Layer.",
    version="1.0.0"
)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Permits rapid local frontend interface integration loops
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    # For local dev, explicitly name the Vite port instead of using "*"
    # allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], 
    # production update
    allow_origins=[
        "http://localhost:5173",
        FRONTEND_URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Thread-Safe Multi-Tenant WebSocket Connection Manager ---
class TenantConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, organization_id: str):
        await websocket.accept()
        if organization_id not in self.active_connections:
            self.active_connections[organization_id] = []
        self.active_connections[organization_id].append(websocket)
        print(f"[WebSocket Engine] Secure link established for Org tenant: {organization_id}")

    def disconnect(self, websocket: WebSocket, organization_id: str):
        if organization_id in self.active_connections:
            self.active_connections[organization_id].remove(websocket)
            if not self.active_connections[organization_id]:
                del self.active_connections[organization_id]
        print(f"[WebSocket Engine] Connection dissolved for Org tenant: {organization_id}")

    async def broadcast_to_tenant(self, organization_id: str, message: dict):
        if organization_id in self.active_connections:
            await asyncio.gather(*[
                ws.send_json(message) for ws in self.active_connections[organization_id]
            ])

stream_manager = TenantConnectionManager()

# --- Authentication Gateway Dependency Filters ---
def get_current_tenant(credentials: HTTPAuthorizationCredentials = Depends(security_bearer)) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        user_id = payload.get("user_id")
        organization_id = payload.get("organization_id")
        if not user_id or not organization_id:
            raise HTTPException(status_code=401, detail="Malformed token claims payload matrix.")
        return {"user_id": user_id, "organization_id": organization_id}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Authentication token signature has expired.")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token signature verification state.")

def parse_ws_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        return {"user_id": payload.get("user_id"), "organization_id": payload.get("organization_id")}
    except Exception:
        return {}

# --- Pydantic Data Structures ---
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    organization_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str
    organization_id: str

class SpendRecord(BaseModel):
    timestamp: str
    cloud_provider: str
    unified_category: str
    cost: float
    is_anomaly: int
    anomaly_type: str

class ForecastRecord(BaseModel):
    date: str
    cloud_provider: str
    unified_category: str
    predicted_cost: float
    lower_bound: float
    upper_bound: float
    model_used: str

class IncidentTrigger(BaseModel):
    date: str
    type: str
    stream: str

class RCALogResponse(BaseModel):
    status: str
    target_date: str
    topology: str
    report: str

# --- 1. Security & Identity Handlers ---

@app.get("/api/v1/health")
def health_check():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database target offline: {str(e)}")

@app.post("/api/v1/auth/signup")
def register_organization_tenant(user_data: UserRegister):
    with engine.begin() as conn:
        existing_user = conn.execute(
            text("SELECT id FROM users WHERE email = :email"), {"email": user_data.email}
        ).fetchone()
        
        if existing_user:
            raise HTTPException(status_code=400, detail="Account registration email already in use.")
        
        org_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        hashed_pass = hash_password(user_data.password)
        
        conn.execute(
            text("INSERT INTO organizations (id, name, tier) VALUES (:id, :name, 'FREE')"),
            {"id": org_id, "name": user_data.organization_name}
        )
        
        conn.execute(
            text("""
                INSERT INTO users (id, organization_id, email, password_hash, first_name, last_name, role)
                VALUES (:id, :org_id, :email, :pass_hash, :f_name, :l_name, 'ADMIN')
            """),
            {
                "id": user_id, "org_id": org_id, "email": user_data.email,
                "pass_hash": hashed_pass, "f_name": user_data.first_name, "l_name": user_data.last_name
            }
        )
            
    return {"status": "success", "message": f"Tenant organization '{user_data.organization_name}' provisioned successfully."}

@app.post("/api/v1/auth/login", response_model=AuthTokenResponse)
def login_user_authenticate(credentials: UserLogin):
    with engine.connect() as conn:
        user = conn.execute(
            text("SELECT id, organization_id, password_hash FROM users WHERE email = :email"),
            {"email": credentials.email}
        ).mappings().fetchone()
        
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid account authorization credentials.")
        
    token = create_access_token({"user_id": str(user["id"]), "organization_id": str(user["organization_id"])})
    return {"access_token": token, "token_type": "bearer", "organization_id": str(user["organization_id"])}

# --- 2. Telemetry Retrieval Analytics & Real-Time Stream Layer ---

@app.get("/api/v1/spend", response_model=List[SpendRecord])
def get_actual_spend(
    provider: str, category: str, start_date: date, end_date: date,
    tenant: dict = Depends(get_current_tenant)
):
    query = text("""
        SELECT timestamp, cloud_provider, unified_category, cost, is_anomaly, anomaly_type
        FROM cloud_spend
        WHERE organization_id = :org_id AND cloud_provider = :provider 
          AND unified_category = :category AND timestamp >= :start_date AND timestamp <= :end_date
        ORDER BY timestamp ASC;
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {
            "org_id": tenant["organization_id"], "provider": provider, 
            "category": category, "start_date": start_date, "end_date": end_date
        }).mappings().all()
    
    return [{**r, "timestamp": str(r["timestamp"])} for r in result]

@app.get("/api/v1/forecast", response_model=List[ForecastRecord])
def get_forecast(tenant: dict = Depends(get_current_tenant)):
    forecast_path = os.path.join(ROOT_DIR, "data", "production_forecast.csv")
    try:
        df = pd.read_csv(forecast_path)
        return df.to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Ensembled prediction matrices missing.")

@app.post("/api/v1/anomaly/rca", response_model=RCALogResponse)
async def trigger_root_cause_analysis(incident: IncidentTrigger, tenant: dict = Depends(get_current_tenant)):
    initial_state = {
        "anomaly_details": {"date": incident.date, "type": incident.type, "stream": incident.stream, "organization_id": tenant["organization_id"]},
        "fetched_data": [], "analysis_notes": [], "final_rca_report": "", "next_step": ""
    }
    try:
        graph_output = await app_graph.ainvoke(initial_state)
        return {
            "status": "success", "target_date": incident.date, "topology": incident.type,
            "report": graph_output.get("final_rca_report", "Analysis phase dropped without report data.")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangGraph Engine Exception: {str(e)}")

@app.websocket("/api/v1/telemetry/stream")
async def secure_telemetry_websocket_stream(websocket: WebSocket, token: str = Query(...)):
    tenant = parse_ws_token(token)
    if not tenant:
        await websocket.close(code=4001)
        return
        
    org_id = tenant["organization_id"]
    await stream_manager.connect(websocket, org_id)
    
    try:
        while True:
            client_heartbeat = await websocket.receive_text()
            await websocket.send_json({"heartbeat": "acknowledged", "server_time": str(datetime.utcnow())})
    except WebSocketDisconnect:
        stream_manager.disconnect(websocket, org_id)
    except Exception as e:
        print(f"[WebSocket Runtime Error] Client boundary broken: {str(e)}")
        stream_manager.disconnect(websocket, org_id)

# --- 3. Interview Demo / Developer Sandbox Routes ---

class DemoTriggerResponse(BaseModel):
    status: str
    message: str
    target_date: str
    cost_injected: float

@app.post("/api/v1/dev/simulate_anomaly", response_model=DemoTriggerResponse)
def trigger_live_interview_demo(tenant: dict = Depends(get_current_tenant)):
    """
    DEV ONLY: Simulates an incoming AWS EventBridge webhook firing a massive 
    cost overrun into the database. Used for live portfolio demonstrations.
    """
    import random
    
    # Target date specifically aligned with our mock dataset timeline
    target_date = "2026-06-28"
    massive_cost = round(random.uniform(15000, 19000), 2)
    org_id = tenant["organization_id"]
    
    query = text("""
        INSERT INTO cloud_spend 
        (organization_id, timestamp, cloud_provider, unified_category, cost, normalized_env, normalized_team, is_anomaly, anomaly_type)
        VALUES 
        (:org_id, :target_date, 'AWS', 'COMPUTE', :cost, 'dev', 'ai-research', 1, 'massive_spike')
    """)
    
    try:
        with engine.begin() as conn:
            conn.execute(query, {
                "org_id": org_id,
                "target_date": target_date,
                "cost": massive_cost
            })
        return {
            "status": "success", 
            "message": "AWS Webhook simulated. Anomaly injected into database.",
            "target_date": target_date,
            "cost_injected": massive_cost
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inject demo anomaly: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)