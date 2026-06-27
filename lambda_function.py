

import json
import boto3
import os

sagemaker_runtime = boto3.client("sagemaker-runtime")


ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
THRESHOLD = float(os.environ["THRESHOLD"])


FEATURE_ORDER = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]


def validate_input(body: dict) -> tuple[bool, str]:
    """
    Check that all required fields are present and numeric.
    Returns (is_valid, error_message).
    """
    missing = [f for f in FEATURE_ORDER if f not in body]
    if missing:
        return False, f"Missing fields: {missing}"

    non_numeric = []
    for f in FEATURE_ORDER:
        try:
            float(body[f])
        except (ValueError, TypeError):
            non_numeric.append(f)
    if non_numeric:
        return False, f"Non-numeric values in fields: {non_numeric}"

    return True, ""


def build_csv_payload(body: dict) -> str:
    """
    Convert the JSON body to a comma-separated string
    in the exact feature order the model expects.
    """
    values = [str(float(body[f])) for f in FEATURE_ORDER]
    return ",".join(values)


def lambda_handler(event, context):
    """
    Main Lambda handler. Called for every API request.

    event: API Gateway passes the HTTP request here as a dict
    context: Lambda metadata (function name, timeout remaining, etc.)
    """

    # --- Handle CORS preflight (OPTIONS request from browsers) ---
    if event.get("httpMethod") == "OPTIONS":
        return cors_response(200, {})

    try:
        # --- Parse request body ---
        # API Gateway passes the body as a JSON string, so we parse it.
        raw_body = event.get("body", "{}")
        if isinstance(raw_body, str):
            body = json.loads(raw_body)
        else:
            body = raw_body   # Already parsed (some integrations do this)

        # --- Validate input ---
        is_valid, error_msg = validate_input(body)
        if not is_valid:
            return cors_response(400, {"error": f"Invalid input: {error_msg}"})

        # --- Build CSV payload ---
        # The SageMaker XGBoost endpoint expects a comma-separated string
        csv_payload = build_csv_payload(body)

        # --- Call SageMaker endpoint ---
        sm_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",   # We're sending CSV
            Accept="text/csv",        # XGBoost returns CSV (a single float)
            Body=csv_payload
        )

        # --- Parse SageMaker response ---
        # Response body is bytes like b"0.8732\n"
        raw_score = sm_response["Body"].read().decode("utf-8").strip()
        fraud_probability = float(raw_score)

        # --- Apply threshold to get binary decision ---
        is_fraud = fraud_probability >= THRESHOLD

        # --- Build risk level ---
        if fraud_probability >= 0.7:
            risk_level = "HIGH"
            action     = "BLOCK"
        elif fraud_probability >= 0.3:
            risk_level = "MEDIUM"
            action     = "REVIEW"
        else:
            risk_level = "LOW"
            action     = "APPROVE"

        # --- Return structured response ---
        result = {
            "fraud_detected"   : is_fraud,
            "fraud_probability": round(fraud_probability, 4),
            "risk_level"       : risk_level,
            "recommended_action": action,
            "threshold_used"   : THRESHOLD,
            "model_endpoint"   : ENDPOINT_NAME
        }

        return cors_response(200, result)

    except json.JSONDecodeError as e:
        return cors_response(400, {"error": f"Invalid JSON: {str(e)}"})

    except sagemaker_runtime.exceptions.ModelError as e:
        # Model returned an error (e.g. wrong number of features)
        return cors_response(422, {"error": f"Model error: {str(e)}"})

    except Exception as e:
        # Log unexpected errors to CloudWatch
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        return cors_response(500, {"error": "Internal server error"})


def cors_response(status_code: int, body: dict) -> dict:
    """
    Wrap response with CORS headers.
    CORS headers allow browsers from any origin to call this API.
    """
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin" : "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        },
        "body": json.dumps(body)
    }