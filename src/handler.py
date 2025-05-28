import json, os, urllib.request, boto3, urllib.parse

s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime")
MODEL_ID = os.getenv("MODEL_ID", "amazon.titan-text-lite-v1")

# ---------- helper -------------------------------------------------
def summarise(text: str) -> str:
    prompt = (
        "You are a helpful assistant.\n"
        "Summarize the following passage in **exactly five** short bullet points.\n"
        "Each point should describe a key event or idea.\n"
        "Only return five bullets. Do not continue the story.\n\n"
        f"{text[:8000]}"
    )
    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {"maxTokenCount": 256, "temperature": 0.2}
    })
    resp = bedrock.invoke_model(modelId=MODEL_ID, body=body)
    return json.loads(resp["body"].read())["results"][0]["outputText"]

# ---------- Lambda entrypoint -------------------------------------
def lambda_handler(event, context):
    # A. S3 OBJECT CREATED trigger (unchanged)
    if "Records" in event and event["Records"][0]["eventSource"].startswith("aws:s3"):
        rec    = event["Records"][0]["s3"]
        bucket = rec["bucket"]["name"]
        key    = urllib.parse.unquote_plus(rec["object"]["key"])
        text   = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8", "ignore")
        summary = summarise(text)
        out_key = f"summary/{os.path.basename(key).rsplit('.',1)[0]}_summary.txt"
        s3.put_object(Bucket=bucket, Key=out_key, Body=summary.encode())
        return {"status": "OK", "wrote": out_key}

    # B. NEW /summarise-text POST  (called from the HTML page)
    if event.get("rawPath") == "/summarise-text":
        body    = json.loads(event.get("body") or "{}")
        text    = body.get("text", "")
        if not text:
            return _cors({"statusCode": 400, "body": "Missing text"})

        summary = summarise(text)
        return _cors({"statusCode": 200, "body": json.dumps({"summary": summary})})

    # C. Fallback
    return _cors({"statusCode": 404, "body": "Not found"})

# ---------- helper to add CORS header ------------------------------
def _cors(resp: dict) -> dict:
    resp.setdefault("headers", {})
    resp["headers"]["Access-Control-Allow-Origin"] = "*"
    resp["headers"]["Access-Control-Allow-Headers"] = "*"
    return resp
