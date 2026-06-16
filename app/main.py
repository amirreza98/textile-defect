import json
import base64
import boto3
import os
import uuid
import glob
from ultralytics import YOLO

model = YOLO("/app/best.pt")

s3 = boto3.client('s3')
BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

def handler(event, context):
    try:
        # Handle both direct and API Gateway calls
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)

        image_data = base64.b64decode(body['image'])
        
        temp_input = f"/tmp/input_{uuid.uuid4()}.jpg"
        
        with open(temp_input, 'wb') as f:
            f.write(image_data)
        
        # Run detection
        results = model.predict(
            source=temp_input,
            conf=0.25,
            save=True,
            project="/tmp",
            name="results",
            exist_ok=True
        )
        
        # Get detections
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    'class': model.names[int(box.cls)],
                    'confidence': float(box.conf),
                })
        
        # Find saved result image
        saved_files = glob.glob("/tmp/results/*.jpg")
        result_image_path = saved_files[0] if saved_files else temp_input
        
        # Upload to S3
        image_key = f"results/{uuid.uuid4()}.jpg"
        s3.upload_file(result_image_path, BUCKET_NAME, image_key)
        
        image_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{image_key}"
        
        # Cleanup
        os.remove(temp_input)
        if saved_files:
            os.remove(saved_files[0])

        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'content-type',
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
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': str(e),
                'detections': [],
                'total_defects': 0
            })
        }