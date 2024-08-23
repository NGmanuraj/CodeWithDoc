import requests
import PyPDF2
from io import BytesIO
from io import StringIO
from bs4 import BeautifulSoup
import aiohttp
import asyncio
import streamlit as st

async def fetch_url(session, url):
    try:
        async with session.get(url) as response:
            response.raise_for_status()  # Raise an exception for non-200 status codes
            return await response.text()
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return None  # Indicate error by returning None

async def load_url_async(url):
    async with aiohttp.ClientSession() as session:
        text = await fetch_url(session, url)
        if text:
            return [{'text': text}]
        else:
            return None

def load_pdf(file):
    pdf_text = ""
    with BytesIO(file.read()) as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            pdf_text += page.extract_text() + "\n"
    return [{'text': pdf_text}]

def load_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        html_content = response.text
        
        # Parse HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract text from the parsed HTML
        text = soup.get_text(separator='\n', strip=True)
        
        # Optional: split text into smaller chunks if needed
        return text
    except requests.RequestException as e:
        st.error(f"Error loading URL: {e}")
        return None

def load_text(file):
    text = file.read().decode("utf-8")
    return [{'text': text}]
