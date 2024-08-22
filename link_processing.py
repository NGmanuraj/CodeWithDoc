# link_processing.py
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
import urllib.parse

def process_link(link):
    parsed_url = urllib.parse.urlparse(link)
    
    if "youtube.com" in parsed_url.netloc or "youtu.be" in parsed_url.netloc:
        return process_youtube_link(link)
    else:
        return process_website_link(link)

def process_youtube_link(link):
    try:
        video_id = extract_youtube_video_id(link)
        if video_id:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([entry['text'] for entry in transcript])
            return transcript_text
        else:
            return "Error: Video ID could not be extracted."
    except Exception as e:  # Catch any general exception
        return f"Error fetching YouTube transcript: {str(e)}"

def extract_youtube_video_id(link):
    parsed_url = urllib.parse.urlparse(link)
    if "youtube.com" in parsed_url.netloc:
        video_id = urllib.parse.parse_qs(parsed_url.query).get('v')
        return video_id[0] if video_id else None
    elif "youtu.be" in parsed_url.netloc:
        return parsed_url.path[1:]
    return None

def process_website_link(link):
    try:
        response = requests.get(link)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n".join([para.get_text() for para in paragraphs])
        return text
    except requests.RequestException as e:
        return f"Error fetching website content: {str(e)}"