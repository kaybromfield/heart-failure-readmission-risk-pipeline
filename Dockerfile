# ============================================
# Dockerfile -- Lambda container image for batch_scoring.py
# ============================================
# Uses AWS's official Python base image for Lambda container support.
# This image is deployed to ECR and run by Lambda on a schedule via
# EventBridge, rather than being run manually from a laptop.
# ============================================

FROM public.ecr.aws/lambda/python:3.11

# Copy dependency list and install into the Lambda task root
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --only-binary=:all: -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy application code
COPY src/model_utils.py ${LAMBDA_TASK_ROOT}
COPY src/lambda_handler.py ${LAMBDA_TASK_ROOT}

# Lambda needs an explicit handler function, not a __main__ block --
# lambda_handler.py wraps run_batch_scoring() for this purpose.
CMD ["lambda_handler.handler"]
