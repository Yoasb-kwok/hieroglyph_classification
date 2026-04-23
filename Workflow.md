Custom Object Detection: Red Box with 人 & Blue Box with 德
Problem
This workflow detects two custom-labeled objects — 'Red Box with 人' and 'Blue Box with 德' — in real-time from webcam and mobile phone feeds. Because these are non-COCO objects, a custom model must be trained using Roboflow Rapid before the workflow can be deployed.

Approach
Train a custom object detection model using Roboflow Rapid on labeled images of the two target classes. Once trained, deploy the model inside a Workflow that runs inference, visualizes results, and continuously uploads new frames with predictions back to the Roboflow dataset to enable an active learning loop for ongoing improvement.

Models
Custom Roboflow Rapid Model: Trained on images labeled with 'Red Box with 人' and 'Blue Box with 德'. Use Roboflow Rapid training (fast training mode) in your Roboflow project. Once trained, note the model_id (e.g., red-blue-box-detector/1) to plug into the workflow.
Workflow Steps
Input: Accept an image (delivered per-frame from webcam or mobile camera stream).

Detect custom objects: Run the custom Roboflow Rapid-trained object detection model on the image, filtering for classes ['Red Box with 人', 'Blue Box with 德'] with a confidence threshold of 0.4. [Object Detection Model]

Draw bounding boxes: Overlay colored bounding boxes around each detected object. Use a custom color palette to distinguish the two classes visually (e.g., red for 'Red Box with 人', blue for 'Blue Box with 德'). [Bounding Box Visualization]

Add labels: Overlay class name and confidence score labels on each bounding box for clear identification in the live feed. [Label Visualization]

Count detections: Compute the total number of detected objects (across both classes) to support downstream monitoring or alerting. [Property Definition]

Upload to dataset for active learning: Upload each incoming image along with its model predictions as pre-annotations to the Roboflow project dataset. This enables continuous improvement — new real-world frames are automatically queued for review and retraining. Set a data_percentage (e.g., 20%) to avoid uploading every frame and manage quota. [Roboflow Dataset Upload]

Outputs:

Annotated image with bounding boxes and class/confidence labels.
Total detection count (both classes combined).
Raw prediction data (class names, bounding box coordinates, confidence scores).
Dataset upload confirmation status and message.
Beyond the Workflow
Model Training (prerequisite):

Create a Roboflow project with object detection type.
Label at least 50–100 images per class ('Red Box with 人', 'Blue Box with 德') using Roboflow Annotate or upload pre-labeled data.
Train using Roboflow Rapid (fast/1-click training) to get a deployable model quickly.
After training, copy the model_id and set it in the Object Detection Model block.
Real-Time Deployment:

For webcam: Use the Roboflow Inference SDK (pip install inference) and point it at a local webcam stream. The SDK handles frame extraction and passes each frame to the Workflow automatically.
For mobile: Use the Roboflow mobile SDK or a hosted API endpoint. The platform accepts individual JPEG frames POSTed to the Workflow API, so any mobile app can call it via HTTP.
Continuous Improvement Loop:

The Dataset Upload block queues uploaded frames in a Roboflow labeling batch.
Periodically review and correct auto-annotations in Roboflow Annotate.
Retrain the model and update the model_id in the workflow to improve accuracy over time.
