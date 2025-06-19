from fastapi import FastAPI

from mamba_ssm.models.config_mamba import MambaConfig

app = FastAPI()

@app.get("/ping")
def ping():
    return {"message": "Mamba SSM is available"}

@app.get("/config")
def get_config():
    config = MambaConfig()
    return config.to_dict()
