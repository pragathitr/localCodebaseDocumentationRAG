print("Testing imports...")

try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers")
except:
    print("✗ sentence-transformers FAILED")

try:
    import faiss
    print("✓ faiss")
except:
    print("✗ faiss FAILED")

try:
    from bs4 import BeautifulSoup
    print("✓ beautifulsoup4")
except:
    print("✗ beautifulsoup4 FAILED")

try:
    import requests
    print("✓ requests")
except:
    print("✗ requests FAILED")

print("\nAll dependencies ready!")