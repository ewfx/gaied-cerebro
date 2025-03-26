import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import asyncio
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from langchain.document_loaders import PyMuPDFLoader, UnstructuredEmailLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Database setup
DATABASE_URL = "postgresql+asyncpg://user:password@localhost/email_db"
engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# Load request type mapping from input file
async def load_request_mapping(file_path="request_mapping.json"):
    async with aiofiles.open(file_path, "r") as f:
        return json.loads(await f.read())

# Load key details mapping from input file
async def load_key_details(file_path="key_details.json"):
    async with aiofiles.open(file_path, "r") as f:
        return json.loads(await f.read())

# Initialize LLM
llm = OpenAI(model_name="gpt-4", temperature=0)
embedding_model = OpenAIEmbeddings()

async def process_email_with_attachments(email_path):
    """Load email and extract attachments separately."""
    loader = UnstructuredEmailLoader(email_path)
    docs = loader.load()
    email_content = docs[0].page_content if docs else ""
    
    attachment_data = []
    for attachment in docs[0].attachments:
        attachment_loader = PyMuPDFLoader(attachment)
        attachment_text = attachment_loader.load()
        attachment_data.append(attachment_text[0].page_content)
    
    return email_content, attachment_data

async def classify_email(content, request_mapping):
    """Classify email based on body content only, returning multiple possible request types."""
    prompt = PromptTemplate(
        template="""
        Classify this email based on the request mapping:
        {email}
        Mapping: {mapping}
        Provide a ranked list of possible request types with confidence scores and reasoning.
        Return format: request_type1 | confidence_score1 | reasoning1 ; request_type2 | confidence_score2 | reasoning2
        """,
        input_variables=["email", "mapping"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = await chain.arun(email=content, mapping=json.dumps(request_mapping))
    classifications = []
    for part in response.split(";"):
        parts = part.split("|")
        if len(parts) >= 3:
            classifications.append({
                "request_type": parts[0].strip(),
                "confidence_score": float(parts[1].strip()),
                "reasoning": parts[2].strip()
            })
    return classifications

async def extract_fields(content, attachments, request_type, key_details):
    """Extract key fields based on the request type from both email body and attachments."""
    details = key_details.get(request_type, {})
    prompt = PromptTemplate(template="Extract key details from: {email}\nDetails: {details}", input_variables=["email", "details"])
    chain = LLMChain(llm=llm, prompt=prompt)
    email_fields = json.loads(await chain.arun(email=content, details=json.dumps(details)))
    
    attachment_fields = {}
    for attachment in attachments:
        attachment_prompt = PromptTemplate(template="Extract numerical details from: {attachment}", input_variables=["attachment"])
        attachment_chain = LLMChain(llm=llm, prompt=attachment_prompt)
        extracted = json.loads(await attachment_chain.arun(attachment=attachment))
        attachment_fields.update(extracted)
    
    email_fields.update(attachment_fields)
    return email_fields

async def detect_duplicates(content):
    """Detect duplicate emails using pgvector in PostgreSQL."""
    vector = await embedding_model.aembed_query(content)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            """
            SELECT id FROM email_results
            ORDER BY embedding <-> CAST(:vector AS vector)
            LIMIT 1;
            """,
            {"vector": vector}
        )
        existing = result.fetchone()
        return existing[0] if existing else None

async def save_to_db(data, content):
    """Save email processing results to PostgreSQL with embeddings."""
    vector = await embedding_model.aembed_query(content)
    async with AsyncSessionLocal() as session:
        async with session.begin():
            session.add(EmailProcessingResult(
                email=data["email"],
                request_type=data["request_type"],
                sub_request_type=data["sub_request_type"],
                confidence_score=data["confidence_score"],
                reasoning=data["reasoning"],
                fields=data["fields"],
                duplicate=data["duplicate"],
                embedding=vector
            ))

async def load_results():
    """Load processed emails from PostgreSQL."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(EmailProcessingResult))
        return result.scalars().all()

# --- Streamlit UI ---
st.title("📧 AI-Powered Email Classification Dashboard")

uploaded_file = st.file_uploader("Upload an Email (EML, PDF, DOCX)", type=["eml", "pdf", "docx"])
if uploaded_file:
    async def process_and_display():
        content, attachments = await process_email_with_attachments(uploaded_file.name)
        request_mapping = await load_request_mapping()
        key_details = await load_key_details()
        classifications = await classify_email(content, request_mapping)
        
        result_list = []
        for classification in classifications:
            extracted_fields = await extract_fields(content, attachments, classification["request_type"], key_details)
            duplicate_id = await detect_duplicates(content)
            result = {
                "email": uploaded_file.name,
                "request_type": classification["request_type"],
                "confidence_score": classification["confidence_score"],
                "reasoning": classification["reasoning"],
                "fields": extracted_fields,
                "duplicate": bool(duplicate_id),
            }
            await save_to_db(result, content)
            result_list.append(result)
        st.success("Email processed successfully!")
        st.json(result_list)
    
    asyncio.run(process_and_display())

# Display results
data = asyncio.run(load_results())
df = pd.DataFrame(data)
st.dataframe(df)
