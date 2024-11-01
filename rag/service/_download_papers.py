import logging
import os
import time

import requests

from rag._utils import get_filename_from_url, sanitize_filename

logger = logging.getLogger(__name__)

# Parameters
search_query = "agent"  # Replace with desired search term or topic
max_results = 5  # Adjust the number of papers you want to download
output_folder = "data"  # Folder to store downloaded papers
base_url = "http://export.arxiv.org/api/query?"


def fetch_arxiv_papers(search_query, max_results=5):
    """Fetches metadata of papers from arXiv using the API."""
    url = f"{base_url}search_query=all:{search_query}&start=0&max_results={max_results}"
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def parse_paper_links(response_text):
    """Parses paper links and titles from arXiv API response XML."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(response_text)
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        pdf_link = None
        for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
            if link.attrib.get("title") == "pdf":
                pdf_link = link.attrib["href"] + ".pdf"
                break
        if pdf_link:
            title = get_filename_from_url(pdf_link)
            print(title)
            papers.append((title, pdf_link))
    return papers


def download_paper(title, pdf_link, output_folder):
    """Downloads a single paper PDF."""
    # Create a safe filename
    safe_title = sanitize_filename(title)
    filename = os.path.join(output_folder, f"{safe_title}.pdf")
    response = requests.get(pdf_link, stream=True)
    response.raise_for_status()

    # Write the PDF to the specified folder
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    print(f"Downloaded: {title}")


def main(search_query, max_results):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Fetch and parse papers
    print(f"Searching for papers on '{search_query}'...")
    response_text = fetch_arxiv_papers(search_query, max_results)
    papers = parse_paper_links(response_text)

    # Download each paper
    print(f"Found {len(papers)} papers. Starting download...")
    for title, pdf_link in papers:
        try:
            download_paper(title, pdf_link, output_folder)
            time.sleep(2)  # Pause to avoid hitting rate limits
        except Exception as e:
            print(f"Failed to download '{title}': {e}")
