import json, os, urllib.parse, urllib.request, boto3

s3       = boto3.client("s3")
bedrock  = boto3.client("bedrock-runtime")
MODEL_ID = os.getenv("MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

# ---------- helper -----------------------------------------------------------
def summarise(text: str) -> str:
    prompt = (
        "You are a helpful assistant.\n"
        "Summarize the following passage in five concise bullet points.\n"
        "Each bullet point should describe a key event or idea.\n"
        "Do not continue the story. Do not repeat information.\n\n"
        "Passage:\n"
        f"{text[:8000]}"
    )
    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 256,
            "temperature": 0.2,
            "topP": 0.9
        }
    })
    resp = bedrock.invoke_model(modelId=MODEL_ID, body=body)
    return json.loads(resp["body"].read())["results"][0]["outputText"]


# ---------- Lambda entrypoint ------------------------------------------------
def lambda_handler(event, context):
    # ── A. called by S3 trigger ────────────────────────────────────────────
    if "Records" in event and event["Records"][0]["eventSource"].startswith("aws:s3"):
        rec    = event["Records"][0]["s3"]
        bucket = rec["bucket"]["name"]
        key    = urllib.parse.unquote_plus(rec["object"]["key"])

        raw    = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8", "ignore")
        summary = summarise(raw)

        out_key = f"summary/{os.path.basename(key).rsplit('.',1)[0]}_summary.txt"
        s3.put_object(Bucket=bucket, Key=out_key, Body=summary.encode())
        return {"status": "OK", "wrote": out_key}

    # ── B. called via HTTP POST /summarise ─────────────────────────────────
    if "body" in event:
        payload = json.loads(event["body"])
    else:
        payload = event  # direct call from AWS CLI

    url = payload.get("url")
    if not url:
        return {"statusCode": 400, "body": "Missing 'url' field"}

    text_bytes = urllib.request.urlopen(url, timeout=10).read()
    text       = text_bytes.decode("utf-8", errors="ignore")
    summary    = summarise(text)

    return {"statusCode": 200, "body": summary}
