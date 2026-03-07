# app.py
# ---------------------------------------------------------
# Required installations before running:
# pip install fastapi uvicorn sqlalchemy pydantic
# ---------------------------------------------------------

import uvicorn
import random
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ==========================================
# 1. DATABASE SETUP (SQLite)
# ==========================================
SQLALCHEMY_DATABASE_URL = "sqlite:///./insurance_fraud.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Table Model
class ClaimRecord(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, index=True)
    policy_number = Column(String, index=True)
    claim_amount = Column(Float)
    incident_severity = Column(String)
    fraud_score = Column(Float)
    is_fraudulent = Column(Boolean)
    flagged_reasons = Column(String)

# Create the database and tables
Base.metadata.create_all(bind=engine)

# ==========================================
# 2. PYDANTIC SCHEMAS (API Input/Output)
# ==========================================
class ClaimRequest(BaseModel):
    policy_number: str
    claim_amount: float
    previous_claims: int
    incident_severity: str  # e.g., "Minor", "Major", "Total Loss"
    police_report_filed: bool
    customer_age: int

class FraudResponse(BaseModel):
    claim_id: int
    fraud_score: float
    is_fraudulent: bool
    flagged_reasons: str

# ==========================================
# 3. FRAUD DETECTION LOGIC
# ==========================================
def evaluate_fraud_risk(claim: ClaimRequest):
    score = 0.0
    reasons =[]

    # --- Rule-Based Checks ---
    if claim.claim_amount > 50000:
        score += 0.3
        reasons.append("Unusually high claim amount")
        
    if claim.previous_claims >= 3:
        score += 0.4
        reasons.append("High frequency of previous claims")
        
    if claim.incident_severity == "Total Loss" and not claim.police_report_filed:
        score += 0.5
        reasons.append("Total loss reported without a police report")

    if claim.customer_age < 21 and claim.claim_amount > 20000:
        score += 0.2
        reasons.append("High claim amount for young driver profile")

    # --- Simulated Machine Learning Model ---
    # (In production, load a .pkl model and use model.predict_proba() here)
    ml_score = random.uniform(0.0, 0.2) 
    score += ml_score

    # Calculate final results
    final_score = min(round(score, 2), 1.0)
    is_fraud = final_score >= 0.70  # Mark as fraud if risk is 70% or higher
    final_reasons = ", ".join(reasons) if reasons else "Normal claim behavior"

    return final_score, is_fraud, final_reasons

# ==========================================
# 4. FASTAPI APP & ROUTES
# ==========================================
app = FastAPI(title="Insurance Fraud Detection API", version="1.0")

# Dependency to open/close DB session per request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/v1/analyze-claim", response_model=FraudResponse)
def analyze_claim(claim: ClaimRequest, db: Session = Depends(get_db)):
    """Receives claim data, runs fraud check, saves to DB, and returns results."""
    
    # 1. Evaluate the claim
    score, is_fraud, reasons = evaluate_fraud_risk(claim)

    # 2. Save result to database
    db_claim = ClaimRecord(
        policy_number=claim.policy_number,
        claim_amount=claim.claim_amount,
        incident_severity=claim.incident_severity,
        fraud_score=score,
        is_fraudulent=is_fraud,
        flagged_reasons=reasons
    )
    db.add(db_claim)
    db.commit()
    db.refresh(db_claim)

    # 3. Return the response
    return FraudResponse(
        claim_id=db_claim.id,
        fraud_score=score,
        is_fraudulent=is_fraud,
        flagged_reasons=reasons
    )

@app.get("/api/v1/claims", response_model=list[FraudResponse])
def get_all_claims(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    """Fetch previously analyzed claims from the database."""
    claims = db.query(ClaimRecord).offset(skip).limit(limit).all()
    return[
        FraudResponse(
            claim_id=c.id, 
            fraud_score=c.fraud_score, 
            is_fraudulent=c.is_fraudulent, 
            flagged_reasons=c.flagged_reasons
        ) for c in claims
    ]

# ==========================================
# 5. SERVER EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Starting Fraud Detection Backend on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)