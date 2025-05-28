# S3 Summariser with AWS Lambda and Bedrock

A serverless text summarization tool using AWS Lambda, Amazon Bedrock (Titan Model), and Amazon S3. Upload a `.txt` file or share a public URL of a long text, and get a 5-point summary generated using an LLM.

---

## Live Demo

**Lambda URL (POST)**: `[YOUR_API_URL]`

**Payload example**:

```json
{
  "url": "https://www.gutenberg.org/cache/epub/11/pg11.txt"
}
```

**S3 Trigger**: Upload `.txt` file to S3 under the `raw/` folder, and the summary will be generated in the `summary/` folder.

---

## Tools & Services Used

* **AWS Lambda**: Runs the summarization logic in Python.
* **Amazon Bedrock**: Uses Titan Text G1 - Lite to generate summaries.
* **Amazon S3**: Stores input and output files.
* **Amazon SAM**: For deploying the Lambda function.

---

## Project Scope

This project demonstrates how to:

* Trigger summarization from either S3 uploads or a URL input
* Generate clean 5-point summaries from long text using a foundation model
* Run entirely on serverless infrastructure (eligible for AWS free tier)

---

## Limitations

* Titan Lite has a 4096 token input limit, so very large texts are truncated.
* Summary quality may vary and needs prompt tuning.
* Summaries overwrite existing files if the same name is uploaded again.

---

## How It Works

1. User uploads a `.txt` file to `s3://s3-summariser-data/raw/`
2. Lambda gets triggered, reads the text, and sends it to Bedrock Titan
3. The model returns a 5-bullet summary
4. Summary is saved as a new `.txt` file in `s3://s3-summariser-data/summary/`

Alternatively:

* You can call the Lambda via API Gateway by passing a public text URL

---

## Run It Yourself

### 1. Clone the repo

```bash
git clone https://github.com/raunakghawghawe/s3-summariser.git
cd s3-summariser
```

### 2. Deploy using AWS SAM

```bash
sam build
sam deploy --guided
```

### 3. Test it

```bash
aws lambda invoke --function-name YOUR_FUNCTION_NAME \
  --cli-binary-format raw-in-base64-out \
  --payload '{"url":"https://www.gutenberg.org/cache/epub/11/pg11.txt"}' \
  response.json && cat response.json
```

---

## GitHub Pages (Optional Web UI)

You can deploy a simple GitHub Page to showcase this project. Ask if you'd like help setting this up.

---

## Author

Built by [Raunak Ghawghawe](https://github.com/raunakghawghawe)

If you're a recruiter, feel free to explore the code, review the architecture, and test the demo to understand how I build scalable, serverless data products with AWS.
