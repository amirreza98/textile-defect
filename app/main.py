import json
import base64
import boto3
import os
import uuid
from ultralytics import YOLO

# Load model once when Lambda starts (not on every request)
model = YOLO("/app/best.pt")

s3 = boto3.client('s3')
BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

def handler(event, context):
    try:
        # Get image from request
        body = json.loads(event['body'])
        image_data = base64.b64decode(body['image'])
        
        # Save temp image
        temp_input = f"/tmp/input_{uuid.uuid4()}.jpg"
        temp_output = f"/tmp/output_{uuid.uuid4()}.jpg"
        
        with open(temp_input, 'wb') as f:
            f.write(image_data)
        
        # Run detection
        results = model.predict(
            source=temp_input,
            conf=0.25,
            save=True,
            project="/tmp",
            name="results"
        )
        
        # Get detection info
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    'class': model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                })
        
        # Upload result image to S3
        result_image_path = f"/tmp/results/{os.path.basename(temp_input)}"
        image_key = f"results/{uuid.uuid4()}.jpg"
        
        s3.upload_file(result_image_path, BUCKET_NAME, image_key)
        
        # Generate public URL
        image_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{image_key}"
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'detections': detections,
                'image_url': image_url,
                'total_defects': len(detections)
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }