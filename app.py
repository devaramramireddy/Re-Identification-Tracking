from fastapi import FastAPI, HTTPException
import traceback
from typing import Optional
from uvicorn import run
from fastapi.responses import JSONResponse
from tracker import PersonTracker
import logging
from config import Config  # Import the TrackerConfig class from tracker_config.py


app = FastAPI()
logging.basicConfig(level=logging.INFO)

@app.post("/process_frames")
def process_frames(config: Config, optional_params: Optional[Config] = None):
    try:
        default_config = Config()

        # Merge default parameters with parameters from the curl request
        merged_config_dict = default_config.dict()
        merged_config_dict.update(config.dict())
        if optional_params:
            merged_config_dict.update(optional_params.dict())

        # Create a new TrackerConfig instance with merged values
        merged_config = Config(**merged_config_dict)

        person_tracker = PersonTracker(merged_config.dict())

        # Process the video and display frames
        person_tracker.process_video()

        return JSONResponse(content={
            "message": "Processing frames completed.",
            # Add your result_output here
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)
