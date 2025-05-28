import json, os, urllib.request, urllib.parse, uuid, datetime
import boto3

s3       = boto3.client("s3")
bedrock  = boto3.client("bedrock-runtime")

BUCKET          = os.environ["BUCKET_NAME"]
DEFAULT_MODELID = os.environ.get("MODEL_ID", "amazon.titan-text-lite-v1")

# ── model mapping ─────────────────────────────────────────────
MODELS = {
    "titan": "amazon.titan-text-lite-v1",
    "haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "cohere-light": "cohere.command-light-text-v14"
}

# ── Bedrock helper ────────────────────────────────────────────
# def _invoke_bedrock(prompt: str, model_id: str) -> str:
#     body = json.dumps({
#         "inputText": prompt,
#         "textGenerationConfig": {"maxTokenCount": 256, "temperature": 0.2}
#     })
#     resp = bedrock.invoke_model(modelId=model_id, body=body)
#     return json.loads(resp["body"].read())["results"][0]["outputText"]

def _invoke_bedrock(payload, model_id: str) -> str:
    if model_id.startswith("anthropic."):
        system_msg = payload["system"]
        user_msg   = payload["user"]
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_msg,
            "messages": [{"role": "user", "content": user_msg}],
            "max_tokens": 256,
            "temperature": 0.2
        })
    elif model_id.startswith("cohere."):
        body = json.dumps({
            "prompt": payload,          # payload is a string here
            "max_tokens": 256,
            "temperature": 0.2,
            "stop_sequences": []
        })
    else:  # Titan / others
        body = json.dumps({
            "inputText": payload,
            "textGenerationConfig": {
                "maxTokenCount": 256,
                "temperature": 0.2
            }
        })

    resp  = bedrock.invoke_model(modelId=model_id, body=body)
    data  = json.loads(resp["body"].read())

    if model_id.startswith("anthropic."):
        return data["content"][0]["text"]
    elif model_id.startswith("cohere."):
        return data["generations"][0]["text"]
    else:
        return data["results"][0]["outputText"]


# def _summarise_chunk(text: str, model: str) -> str:
#     prompt = (
#         "You are a helpful assistant.\n"
#         "Summarise the following passage in short, clear bullet points.\n"
#         "Return **exactly five** concise bullets.\n\n"
#         f"{text}"
#     )
#     return _invoke_bedrock(prompt, model)

def _summarise_chunk(text: str, model: str) -> str:
    if model.startswith("anthropic."):        # Claude path → use (system, user)
        return _invoke_bedrock(
            {
                "system": (
                    "You are a helpful assistant. Summarise the user-provided "
                    "passage in **exactly five** short, clear bullet points."
                ),
                "user": text
            },
            model
        )
    else:                                     # Titan / Cohere keep old prompt
        prompt = (
            "Summarise the following passage in five concise bullet points.\n\n"
            f"{text}"
        )
        return _invoke_bedrock(prompt, model)

def summarise(text: str, model_id: str) -> str:
    chunks  = [text[i:i+4800] for i in range(0, len(text), 4800)][:4]
    bullets = [_summarise_chunk(c, model_id) for c in chunks]
    return _summarise_chunk(" ".join(bullets), model_id)

# ── misc helpers ──────────────────────────────────────────────
def _cors(resp: dict) -> dict:
    resp.setdefault("headers", {})
    resp["headers"]["Access-Control-Allow-Origin"]  = "*"
    resp["headers"]["Access-Control-Allow-Headers"] = "*"
    return resp

def _s3_key(prefix: str, ext: str) -> str:
    ts  = datetime.datetime.utcnow().strftime("%Y/%m/%d/")
    uid = uuid.uuid4().hex
    return f"{prefix}/{ts}{uid}.{ext}"

# ── Lambda entrypoint ─────────────────────────────────────────
def lambda_handler(event, _):
    # model requested by caller (defaults to Titan)
    model_key = json.loads(event.get("body") or "{}").get("model") if "body" in event else event.get("model")
    model_id  = MODELS.get((model_key or "titan").lower(), DEFAULT_MODELID)

    # ── 1) S3 trigger branch (optional) ───────────────────────
    if "Records" in event and event["Records"][0]["eventSource"].startswith("aws:s3"):
        rec      = event["Records"][0]
        bucket   = rec["s3"]["bucket"]["name"]
        key      = urllib.parse.unquote_plus(rec["s3"]["object"]["key"])
        text     = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8", "ignore")
        summary  = summarise(text, DEFAULT_MODELID)
        s3.put_object(
            Bucket=bucket,
            Key=key.replace("raw/", "summary/", 1),
            Body=summary.encode("utf-8"),
            ContentType="text/plain"
        )
        return

    # ── 2) POST /summarise-text  (front-end) ─────────────────
    if event.get("rawPath") == "/summarise-text":
        text = json.loads(event["body"]).get("text", "")

        # 2-a) Write raw text to S3
        raw_key = _s3_key("raw", "txt")
        s3.put_object(
            Bucket=BUCKET,
            Key=raw_key,
            Body=text.encode("utf-8"),
            ContentType="text/plain"
        )

        # 2-b) Generate summary
        summary = summarise(text, model_id)

        # 2-c) Store summary back to S3
        sum_key = raw_key.replace("raw/", "summary/", 1)
        s3.put_object(
            Bucket=BUCKET,
            Key=sum_key,
            Body=summary.encode("utf-8"),
            ContentType="text/plain"
        )

        return _cors({"statusCode": 200,
                      "body": json.dumps({"summary": summary})})

    # ── 3) POST /summarise  (URL input) ───────────────────────
    if event.get("rawPath") == "/summarise":
        url  = json.loads(event["body"]).get("url", "")
        text = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "ignore")
        summary = summarise(text, model_id)
        return _cors({"statusCode": 200,
                      "body": json.dumps({"summary": summary})})

    # ── 4) CLI invoke convenience ─────────────────────────────
    if "text" in event or "url" in event:
        raw = event.get("text") or urllib.request.urlopen(event["url"], timeout=60).read().decode("utf-8", "ignore")
        return _cors({"statusCode": 200,
                      "body": json.dumps({"summary": summarise(raw, model_id)})})

    # ── 5) 404 fallback ───────────────────────────────────────
    return _cors({"statusCode": 404, "body": "Not found"})
