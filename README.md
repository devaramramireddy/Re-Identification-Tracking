# Person Re-Identification and Tracking.

> ## Introductoin
> This project presents an advanced person tracking system that combines the strengths of computer vision and machine learning. It's designed to accurately identify, track, and analyze individuals in video streams using state-of-the-art object detection and tracking algorithms. Each person detected in the video is assigned a unique ID, enabling the system to focus on and track specific individuals upon request. This feature is particularly useful for applications where individual tracking across different frames or scenes is crucial.

> ## System Architecture
> ### Configuration Management (config.py)
> Manages essential system configurations, including model paths, video source, processing options, and image enhancement settings.
> 
> ### API Server (app.py)
> A RESTful API using FastAPI for video frame processing. It listens for video processing requests and initializes the PersonTracker with configurations.
> 
> ### Person Tracking Core (tracker.py)
> The core module for person detection, tracking, and analysis. It utilizes YOLO for object detection, an ReID model for person feature extraction of detected person., and Kalman Filters for movement prediction.
> 
> ### Preprocessing Module (preprocessing.py)
> Enhances video frames for better detection and tracking accuracy. Techniques include image resizing, padding, color normalization, noise reduction, and histogram equalization.


## Project Structure.
```commandline
Person Re-Identification and Tracking
    - data
        - input_videos
    - models
        - yolov5su
        - ReID_model
    - Results
        - Final Processed Video
    - __init__.py
    - config.py
    - app.py
    - tracker.py
    - preprocessing.py
    - requirements.sh
    - Dockerfile
```
>## Installation
> ```commandline
> cd /path/of/the/project
> sh ./requirements.sh
> ```

> ## Execution:
> ### Without Docker:
> ```commandline
> python3 app.py
> ``` 
> 
> ### Using Docker:
>
> sudo docker load -i docker_image_ReID_and_tracker.tar
> > If you want read the input video and other files from docker file please use the following command to run docker container.
> > - sudo docker run -p 8000:8000 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e XAUTHORITY=$XAUTHORITY person_tracker:0.0.1
> 
> > If you want load the input video of other files from local machine please use the following command to run docker container.
> > - sudo docker run -p 8000:8000 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /path/of/the/local/machine/:/mnt -e XAUTHORITY=$XAUTHORITY person_tracker:0.0.1

> 
> ## Curl Request
> - After starting the application(INFO: Application startup complete.), initiate a curl request from a new terminal to interact with the API. 
> - Default parameters are specified in the configuration section. If adjustments are needed, pass the modified parameters through the curl request. 
> - Below are examples of curl requests for both RGB and Thermal images, covering both Non-clustering and Clustering approaches.
> - System automatically display the processed video, If you want the quit the process please press "q" key.
>

> #### Curl Request - send file from local machine(if you run demo from your local python interpreter).
> ```commandline
> curl -X POST http://localhost:8000/process_frames -H "Content-Type: application/json" -d '{
>   "config": {
>     "yolo_model_path": "./models/yolov5su.pt",
>     "reid_model_name": "osnet_ain_x1_0",
>     "reid_model_path": "./models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth",
>     "input_type": "RGB",                                      # Type of Input "RGB" or "THERMAL"
>     "video_path": "./data/Person_counting_assignment.mp4",    # Input Video Path
>     "similarity_threshold_yolo": 0.8,                         # Person Detection Threshold
>     "similarity_threshold_reid": 0.7,                         # ReID Similarity Threshold
>     "tracking_id": 1,                                         # Person ID need to be tracked
>     "output_path": "./Results"                                # Path to save the processed vieo
>   }
> }'
> ```
>
> #### Curl Request files from with in the docker image(if you run demo from docker python interpreter).
> ```commandline
> curl -X POST http://localhost:8000/process_frames -H "Content-Type: application/json" -d '{
>   "config": {
>     "yolo_model_path": "/mnt/yolov5su.pt",
>     "reid_model_name": "osnet_ain_x1_0",
>     "reid_model_path": "/mnt/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth",
>     "input_type": "RGB",                                      # Type of Input "RGB" or "THERMAL"
>     "video_path": "/mnt/Person_counting_assignment.mp4",    # Input Video Path
>     "similarity_threshold_yolo": 0.8,                         # Person Detection Threshold
>     "similarity_threshold_reid": 0.7,                         # ReID Similarity Threshold
>     "tracking_id": 1,                                         # Person ID need to be tracked
>     "output_path": "./Results"                                # Path to save the processed vieo
>   }
> }'
> ```

