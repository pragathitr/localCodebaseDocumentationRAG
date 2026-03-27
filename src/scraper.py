"""
Scrapes FastAPI tutorial docs and converts to Markdown
"""
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
import json
import os


def scrape_fastapi_docs():
    base_url = "https://fastapi.tiangolo.com"

    sections = [
        "/tutorial/first-steps/",
        "/tutorial/path-params/",
        "/tutorial/query-params/",
        "/tutorial/body/",
        "/tutorial/query-params-str-validations/",
        "/tutorial/path-params-numeric-validations/",
        "/tutorial/query-param-models/",
        "/tutorial/body-multiple-params/",
        "/tutorial/body-fields/",
        "/tutorial/body-nested-models/",
        "/tutorial/schema-extra-example/",
        "/tutorial/extra-data-types/",
        "/tutorial/cookie-params/",
        "/tutorial/header-params/",
        "/tutorial/response-status-code/",
        "/tutorial/request-forms/",
        "/tutorial/request-form-models/",
        "/tutorial/request-files/",
        "/tutorial/handling-errors/",
        "/tutorial/cors/",
        "/tutorial/security/",
        "/tutorial/middleware/",
        "/tutorial/sql-databases/",
        "/tutorial/background-tasks/",
        "/tutorial/static-files/",
        "/tutorial/testing/",
        "/tutorial/debugging/",
    ]

    documents = [] #list to store dictionaries of scraped tutorial docs with url, title, and content

    for section in sections:
        url = base_url + section
        print(f"Scraping: {url}")

        try:
            response = requests.get(url)            #response object containing requested html content
            soup = BeautifulSoup(response.content, 'html.parser')           #parses the raw html bytes into a tree structure

            main = soup.find('main')            #returns the <main> element object or None if not found
            if main:                            #if <main> found then convert the html content into markdown
                # Convert HTML to Markdown
                markdown_content = markdownify(
                    str(main),
                    heading_style="ATX",  # Use ## style headings
                    code_language="python"  # Default code language
                )

                documents.append({                  #adding the scraped document as a dictionary to our list
                    'url': url,
                    'title': soup.title.string if soup.title else section,
                    'content': markdown_content
                })
        except Exception as e:
            print(f"Error scraping {url}: {e}")      #error handling for any link that is not able to scraped

    os.makedirs('data/raw', exist_ok=True)
    with open('data/raw/docs.json', 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Scraped {len(documents)} documents")
    print(f"✓ Converted to Markdown")
    print(f"✓ Saved to data/raw/docs.json")


if __name__ == "__main__":
    scrape_fastapi_docs()