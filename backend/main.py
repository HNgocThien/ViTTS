from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.train import router as train_router
from app.api.infer import router as infer_router
from app.api.collect import router as collect_router
from config_loader import API_CONFIG, MODELS
import sys

for m in MODELS.values():
    src_path = m.get("src")
    if src_path and src_path not in sys.path:
        sys.path.insert(0, src_path)

app = FastAPI(title="TTS-Platform Unified API", version=API_CONFIG.get("version", "1.2.0"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(train_router, prefix="/api/train", tags=["Train"])
app.include_router(infer_router, prefix="/api/infer", tags=["Infer"])
app.include_router(collect_router, prefix="/api/collect", tags=["Collect"])

@app.get("/")
async def root():
    return {"message": "TTS Platform API is running", "version": API_CONFIG.get("version", "1.2.0")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=API_CONFIG.get("port", 8000), reload=True)
