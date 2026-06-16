FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies
RUN yum install -y libxcb mesa-libGL

# Install Python dependencies
RUN pip install ultralytics boto3 --no-cache-dir

# Copy model and code
COPY best.pt /app/best.pt
COPY main.py ${LAMBDA_TASK_ROOT}

# Set handler
CMD ["main.handler"]