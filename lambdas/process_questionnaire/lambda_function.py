from aws_lambda_powertools import Logger
import boto3
from botocore.config import Config
import json
import logging
import math
import os
import traceback
import csv
import io
import pandas as pd
import json

lambda_client = boto3.client('lambda')
s3 = boto3.client('s3')

logger = logging.getLogger(__name__)
if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

cloudwatch_logger = Logger()

s3_bucket = os.environ.get('S3_BUCKET', "hackathon-s3-bucket-29-02-2024-tenant-data")
bedrock_region = os.environ.get("BEDROCK_REGION", "us-east-1")
bedrock_url = os.environ.get("BEDROCK_URL", "https://bedrock-runtime.us-east-1.amazonaws.com")
iam_role = os.environ.get("IAM_ROLE", None)


class BedrockInference:
    def __init__(self, bedrock_client, model_id):
        self.bedrock_client = bedrock_client
        self.model_id = model_id

    def invoke_text(self, body):
        try:
            provider = self.model_id.split(".")[0]

            # request_body = LLMInputOutputAdapter.prepare_input(provider, body["inputs"], model_kwargs)

            # request_body = json.dumps(request_body)

            # Titan
            # request_body = json.dumps({
            #     "inputText": body["inputs"],
            #     "textGenerationConfig":{
            #         "maxTokenCount":4096,
            #         "stopSequences":[],
            #         "temperature":0,
            #         "topP":0.9
            #         }
            #     })

            # Claude
            request_body = json.dumps({
                "prompt": body["inputs"],
                "max_tokens_to_sample": 4096,
                "stop_sequences": ["###"],
                "temperature": 0,
                "top_p": 0.9
            })

            response = self.bedrock_client.invoke_model(
                body=request_body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json"
            )

            logger.info(f"reponse = {response}")

            response = json.loads(response.get('body').read())

            logger.info(f"reponse[body] = {response}")

            # Titan
            # response = response.get('results')[0].get('outputText')

            # Claude
            response = response.get('completion')

            logger.info(f"reponse after parsing = {response}")

            return response
        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e


def _get_bedrock_client():
    try:
        logger.info(f"Create new client\n  Using region: {bedrock_region}")
        session_kwargs = {"region_name": bedrock_region}
        client_kwargs = {**session_kwargs}

        retry_config = Config(
            region_name=bedrock_region,
            retries={
                "max_attempts": 10,
                "mode": "standard",
            },
        )
        session = boto3.Session(**session_kwargs)

        if iam_role is not None:
            logger.info(f"Using role: {iam_role}")
            sts = session.client("sts")

            response = sts.assume_role(
                RoleArn=str(iam_role),  #
                RoleSessionName="amazon-bedrock-assume-role"
            )

            client_kwargs = dict(
                aws_access_key_id=response['Credentials']['AccessKeyId'],
                aws_secret_access_key=response['Credentials']['SecretAccessKey'],
                aws_session_token=response['Credentials']['SessionToken']
            )

        if bedrock_url:
            client_kwargs["endpoint_url"] = bedrock_url

        bedrock_client = session.client(
            service_name="bedrock-runtime",
            config=retry_config,
            **client_kwargs
        )

        logger.info("boto3 Bedrock client successfully created!")
        logger.info(bedrock_client._endpoint)
        return bedrock_client

    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error(stacktrace)

        raise e


def _get_tokens(string):
    logger.info("Counting approximation tokens")

    return math.floor(len(string) / 4)


def bedrock_handler(event):
    logger.info("Bedrock Endpoint")

    model_id = event["queryStringParameters"]['model_id']

    team_id = event["headers"]["team_id"]

    bedrock_client = _get_bedrock_client()

    logger.info(event)

    custom_request_id = event["queryStringParameters"]['requestId'] if 'requestId' in event[
        "queryStringParameters"] else None

    bedrock_inference = BedrockInference(bedrock_client, model_id)

    request_id = event['requestContext']['requestId']
    streaming = event["headers"]["streaming"] if "streaming" in event["headers"] else "false"

    logger.info(f"Model ID: {model_id}")
    logger.info(f"Request ID: {request_id}")

    body = json.loads(event["body"])

    logger.info(f"Input body: {body}")

    model_kwargs = body["parameters"] if "parameters" in body else {}

    logger.info("Request type: text")

    # response = bedrock_inference.invoke_text(body, model_kwargs)
    transform_data(bedrock_inference, team_id)

    results = {"statusCode": 200, "body": json.dumps([{"export": "successfully"}])}

    logs = {
        "team_id": team_id,
        "requestId": request_id,
        "region": bedrock_region,
        "model_id": model_id,
        "inputTokens": 0,  # needs to be added
        "outputTokens": 0,  # needs to be added
        "height": None,
        "width": None,
        "steps": None
    }

    cloudwatch_logger.info(logs)

    return results


def llm(bedrock_inference, text):
    prompt = """\n\nHuman: Please classify the below comment with respect to following categories and pick the most suiting category with a confidence score. The potential categories are: Communication, Management, Work Environment & Other. If there is no matching category pick as category Other. 
    Structure your answer in a JSON format including a field for class and one for confidence. 
    The result should be always in a JSON format never only text.

    Here is an example: 

    Text: 
    I feel uncomfortable in the office environment.
    Result:
    { "class": "Work Environment", "confidence": 0.9 }

    Now classify this text and just answer with the JSON result.
    """

    prompt = prompt + "Text: \n" + text + "\n\n" + "Assistant:"

    body = {
        "inputs": prompt,
        "parameters": {
            "maxTokenCount": 4096,
            "temperature": 0.1
        }
    }
    result = bedrock_inference.invoke_text(body)
    logger.info(f"llm result is = {result}")

    result = result.replace("```tabular-data-json", "").rstrip("```")
    result = result.replace("```", "")

    logger.info(f"cleaned llm result is = {result}")

    return result;


def llm_enrichment(bedrock_inference, text):
    result = llm(bedrock_inference, text)
    logger.info(result)
    # Parse JSON string to dict
    try:
        result_dict = json.loads(result)
    except:
        result_dict = {
            "class": result,
            "confidence": "0.0"
        }

    return result_dict


def download_s3(bucket, object):
    return s3.get_object(Bucket=bucket, Key=object)


def upload_s3(body, bucket, object):
    return s3.put_object(Body=body, Bucket=bucket, Key=object)


def transform_data(bedrock_inference, tenant):
    # Download CSV from S3
    obj = s3.get_object(Bucket=s3_bucket, Key=f'{tenant}/RAW/mini-data.csv')
    csv_body = obj['Body'].read().decode('utf-8')
    # Parse CSV
    df = pd.read_csv(io.StringIO(csv_body))
    # Initialize empty list

    print(df)

    new_rows = []
    # Iterate rows and run enrichment
    for index, row in df.iterrows():
        uid = row['uid']
        free_text = row['v_79']

        enrichment_result = llm_enrichment(bedrock_inference, free_text)
        # Extract values from result
        classification = enrichment_result["class"]
        confidence = enrichment_result["confidence"]

        # Add to new rows
        new_rows.append(
            {'uid': uid, 'free_text': free_text, 'classification': classification, 'confidence': confidence})

    # Write new CSV
    csv_string = io.StringIO()
    csv_writer = csv.DictWriter(csv_string, fieldnames=['uid', 'free_text', 'classification', 'confidence'],
                                delimiter='|')
    csv_writer.writeheader()
    csv_writer.writerows(new_rows)

    # Upload new CSV to S3
    upload_s3(csv_string.getvalue(), s3_bucket, f'{tenant}/CLASSIFICATION/output.csv')


def lambda_handler(event, context):
    try:
        return bedrock_handler(event)

    except Exception as e:
        stacktrace = traceback.format_exc()

        logger.error(stacktrace)
        return {"statusCode": 500, "body": json.dumps([{"generated_text": stacktrace}])}
