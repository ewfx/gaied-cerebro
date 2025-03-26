# AI-Powered Email Classification Pipeline

## Overview

This project classifies emails into predefined request types and subtypes, extracts key fields from both email bodies and attachments, detects duplicate emails, and stores structured results in PostgreSQL with pgvector for efficient retrieval.

## Features

- **Request Type Classification:** Identifies multiple possible request types and assigns confidence scores.
- **Multi-Intent Handling:** Supports emails with multiple request types, prioritizing the most relevant ones.
- **Customizable Extraction:** Extracts configurable fields based on the request type from both email content and attachments.
- **Prioritization Rules:** Email body is prioritized over attachments for request type classification, while numerical fields are extracted from attachments.
- **Duplicate Detection:** Uses vector embeddings to identify similar or duplicate emails.
- **Explainability:** Stores LLM’s reasoning for classifications and extracted fields.
- **Scalability:** Supports async processing and PostgreSQL storage for handling large datasets efficiently.
- **Performance Optimization:** Profiling ensures efficient processing for large email volumes.

## Example Input Files

### Request Mapping (request\_mapping.json)

```json
{
  "Loan Request": ["loan application", "new loan", "loan approval"],
  "Payment Inquiry": ["payment status", "transaction details", "billing issue"],
  "Account Update": ["change address", "update contact", "modify details"]
}
```

### Key Details (key\_details.json)

```json
{
  "Loan Request": {"deal_name": "string", "amount": "float", "expiration_date": "date"},
  "Payment Inquiry": {"transaction_id": "string", "payment_amount": "float", "due_date": "date"},
  "Account Update": {"customer_id": "string", "new_address": "string", "phone_number": "string"}
}
```

## Example Emails

### Email with Multiple Intents

**Subject:** Loan Status and Payment Confirmation\
**Body:** "I applied for a loan last week and want to check the status. Also, I made a payment yesterday but haven't received confirmation. Can you help?"

### Attachments

- Loan document with amount: **$250,000**
- Payment receipt showing **$1,500**

## Example Output (Results in Tabular Format)

| Email            | Request Type    | Confidence Score | Prioritization | Extracted Fields                                                                    | Duplicate |
| ---------------- | --------------- | ---------------- | -------------- | ----------------------------------------------------------------------------------- | --------- |
| loan\_status.eml | Loan Request    | 0.95             | High           | {"deal\_name": "Home Loan", "amount": 250000, "expiration\_date": "2025-06-30"}     | No        |
| loan\_status.eml | Payment Inquiry | 0.85             | Medium         | {"transaction\_id": "TXN12345", "payment\_amount": 1500, "due\_date": "2024-04-15"} | No        |

## How Prioritization Works

- **Request Type Prioritization:** Loan-related queries have a higher priority in classification than payment inquiries based on business logic.
- **Content vs Attachment:** Request types are identified from the email body, while numerical details (amount, payment info) are extracted from attachments.

## Performance Benchmarking

### Runtime Analysis

| Step                              | Average Time |
|-----------------------------------|--------------|
| Email Processing (Body & Attachments) | 50-200ms    |
| Classification with LLM           | 1-3s        |
| Field Extraction                  | 1-3s        |
| Duplicate Detection (pgvector)     | 10-50ms     |
| Database Storage (PostgreSQL)      | 10-100ms    |

### Efficiency Enhancements

- **Async Processing:** Email classification, field extraction, and database storage run concurrently.
- **Batch Processing:** Extendable for parallel email handling via event-driven architecture.
- **Vectorized Search:** pgvector improves duplicate detection efficiency.
- **Auto-scaling Support:** Compatible with Azure Functions/Kubernetes for large-scale workloads.

## Running the Pipeline

Ensure PostgreSQL and pgvector are set up, then run:

```bash
streamlit run app.py
```

This will start the UI for uploading emails and displaying classified results in a structured format.
