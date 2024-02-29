from aws_lambda_powertools import Logger
import boto3
from botocore.config import Config
import json
from langchain_community.llms.bedrock import LLMInputOutputAdapter
import logging
import math
import os
import traceback

lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')

logger = logging.getLogger(__name__)
if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

cloudwatch_logger = Logger()

bedrock_region = os.environ.get("BEDROCK_REGION", "us-east-1")
bedrock_url = os.environ.get("BEDROCK_URL", "https://bedrock-runtime.us-east-1.amazonaws.com")
iam_role = os.environ.get("IAM_ROLE", None)


class BedrockInference:
    def __init__(self, bedrock_client, model_id):
        self.bedrock_client = bedrock_client
        self.model_id = model_id

    def invoke_embeddings(self, body, model_kwargs):
        try:
            provider = self.model_id.split(".")[0]

            if provider == "cohere":
                if "input_type" not in model_kwargs.keys():
                    model_kwargs["input_type"] = "search_document"
                if isinstance(body["inputs"], str):
                    body["inputs"] = [body["inputs"]]

                request_body = {**model_kwargs, "texts": body["inputs"]}
            else:
                request_body = {**model_kwargs, "inputText": body["inputs"]}

            request_body = json.dumps(request_body)

            response = self.bedrock_client.invoke_model(
                body=request_body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())

            if provider == "cohere":
                response = response_body.get("embeddings")[0]
            else:
                response = response_body.get("embedding")

            return response
        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e

    def invoke_embeddings_image(self, body, model_kwargs):
        try:
            provider = self.model_id.split(".")[0]

            request_body = {**model_kwargs, "inputImage": body["inputs"]}

            request_body = json.dumps(request_body)

            response = self.bedrock_client.invoke_model(
                body=request_body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())

            if provider == "cohere":
                response = response_body.get("embeddings")[0]
            else:
                response = response_body.get("embedding")

            return response
        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e

    def invoke_image(self, body, model_kwargs):
        try:
            provider = self.model_id.split(".")[0]

            if provider == "stability":
                request_body = {**model_kwargs, "text_prompts": body["text_prompts"]}

                height = model_kwargs["height"] if "height" in model_kwargs else 512
                width = model_kwargs["width"] if "width" in model_kwargs else 512
                steps = model_kwargs["steps"] if "steps" in model_kwargs else 50
            else:
                request_body = {**model_kwargs, "textToImageParams": body["textToImageParams"]}

                height = model_kwargs["imageGenerationConfig"]["height"] if "height" in model_kwargs[
                    "imageGenerationConfig"] else 512
                width = model_kwargs["imageGenerationConfig"]["width"] if "width" in model_kwargs[
                    "imageGenerationConfig"] else 512

                if "quality" in model_kwargs["imageGenerationConfig"]:
                    if model_kwargs["imageGenerationConfig"]["quality"] == "standard":
                        steps = 50
                    else:
                        steps = 51
                else:
                    steps = 50

            request_body = json.dumps(request_body)

            response = self.bedrock_client.invoke_model(
                body=request_body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())

            if provider == "stability":
                response = {"artifacts": response_body.get("artifacts")}
            else:
                response = {"images": response_body.get("images")}

            return response, height, width, steps
        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e

    def invoke_text(self, body, model_kwargs):
        try:
            provider = self.model_id.split(".")[0]

            request_body = LLMInputOutputAdapter.prepare_input(provider, body["inputs"], model_kwargs)

            request_body = json.dumps(request_body)

            response = self.bedrock_client.invoke_model(
                body=request_body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json"
            )

            logger.info(f"reponse = {response}")

            response = LLMInputOutputAdapter.prepare_output(provider, response)

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

    ## Check for embeddings or image
    if "type" in event["headers"]:
        if event["headers"]["type"].lower() == "embeddings":
            embeddings = True
            embeddings_image = False
            image = False
        elif event["headers"]["type"].lower() == "embeddings-image":
            embeddings = False
            embeddings_image = True
            image = False
        elif event["headers"]["type"].lower() == "image":
            embeddings = False
            embeddings_image = False
            image = True
        else:
            embeddings = False
            embeddings_image = False
            image = False
    else:
        embeddings = False
        embeddings_image = False
        image = False

    logger.info(f"Model ID: {model_id}")
    logger.info(f"Request ID: {request_id}")

    body = json.loads(event["body"])

    logger.info(f"Input body: {body}")

    model_kwargs = body["parameters"] if "parameters" in body else {}

    if embeddings:
        logger.info("Request type: embeddings")

        response = bedrock_inference.invoke_embeddings(body, model_kwargs)

        results = {"statusCode": 200, "body": json.dumps([{"embedding": response}])}

        logs = {
            "team_id": team_id,
            "requestId": request_id,
            "region": bedrock_region,
            "model_id": model_id,
            "inputTokens": _get_tokens(body["inputs"]),
            "outputTokens": _get_tokens(response),
            "height": None,
            "width": None,
            "steps": None
        }

        cloudwatch_logger.info(logs)
    elif embeddings_image:
        logger.info("Request type: embeddings-image")

        response = bedrock_inference.invoke_embeddings_image(body, model_kwargs)

        results = {"statusCode": 200, "body": json.dumps([{"embedding": response}])}

        logs = {
            "team_id": team_id,
            "requestId": request_id,
            "region": bedrock_region,
            "model_id": model_id + "-image",
            "inputTokens": _get_tokens(body["inputs"]),
            "outputTokens": _get_tokens(response),
            "height": None,
            "width": None,
            "steps": None
        }

        cloudwatch_logger.info(logs)
    elif image:
        logger.info("Request type: image")

        response, height, width, steps = bedrock_inference.invoke_image(body, model_kwargs)

        results = {"statusCode": 200, "body": json.dumps([response])}

        logs = {
            "team_id": team_id,
            "requestId": request_id,
            "region": bedrock_region,
            "model_id": model_id,
            "inputTokens": None,
            "outputTokens": None,
            "height": height,
            "width": width,
            "steps": steps
        }

        cloudwatch_logger.info(logs)
    else:
        logger.info("Request type: text")

        response = bedrock_inference.invoke_text(body, model_kwargs)

        results = {"statusCode": 200, "body": json.dumps([{"generated_text": response["text"]}])}

        logs = {
            "team_id": team_id,
            "requestId": request_id,
            "region": bedrock_region,
            "model_id": model_id,
            "inputTokens": _get_tokens(body["inputs"]),
            "outputTokens": _get_tokens(response["text"]),
            "height": None,
            "width": None,
            "steps": None
        }

        cloudwatch_logger.info(logs)

    return results


def lambda_handler(event, context):
    try:
        return bedrock_handler(event)

    except Exception as e:
        stacktrace = traceback.format_exc()

        logger.error(stacktrace)
        return {"statusCode": 500, "body": json.dumps([{"generated_text": stacktrace}])}
