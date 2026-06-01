import sys
print(f"Python: {sys.version}")
failures = []

checks = [
    ("arxiv",                   "import arxiv"),
    ("fitz (PyMuPDF)",          "import fitz"),
    ("langchain_groq",          "from langchain_groq import ChatGroq"),
    ("langchain_google_genai",  "from langchain_google_genai import ChatGoogleGenerativeAI"),
    ("rank_bm25",               "from rank_bm25 import BM25Okapi"),
    ("qdrant_client",           "from qdrant_client import QdrantClient"),
    ("neo4j",                   "from neo4j import GraphDatabase"),
    ("langgraph",               "from langgraph.graph import StateGraph, END"),
    ("langchain_core.messages", "from langchain_core.messages import HumanMessage, SystemMessage"),
    ("langchain_core.pydantic", "from langchain_core.pydantic_v1 import BaseModel, Field"),
    ("tenacity",                "from tenacity import retry, stop_after_attempt, wait_exponential"),
    ("tqdm",                    "from tqdm import tqdm"),
    ("structlog",               "import structlog"),
    ("dotenv",                  "from dotenv import load_dotenv"),
    ("difflib (stdlib)",        "import difflib"),
    ("typing_extensions",       "from typing_extensions import TypedDict"),
    ("numpy",                   "import numpy"),
    ("accelerate",              "import accelerate"),
    # Note: sentence_transformers and transformers load very slowly (CUDA/DLL init)
    # They are verified by pip show below, not by import here.
]

for name, stmt in checks:
    try:
        exec(stmt)
        print(f"  OK  {name}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        failures.append(name)

print()
if failures:
    print(f"FAILED imports: {failures}")
    sys.exit(1)
else:
    print("All imports OK.")
