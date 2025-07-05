import os
import json
import asyncio
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from openai import RateLimitError
from langchain_core.runnables.base import Runnable

from .state import GraphState
from .tools import WebSearchTool
from .logger import get_logger

# Get the logger
logger = get_logger()

# --- Pydantic Models for LLM Outputs ---

class GenerateQueriesOutput(BaseModel):
    """Output model for the query generation node."""
    queries: List[str] = Field(
        description="A list of 3-5 distinct search engine queries to find information related to the user's question."
    )

class ReflectOutput(BaseModel):
    """Output model for the reflection node."""
    need_more: bool = Field(
        description="Whether more information is needed to answer the question comprehensively."
    )
    new_queries: Optional[List[str]] = Field(
        description="1-3 new, refined search queries if more information is needed. Can be empty."
    )

class SynthesizeOutput(BaseModel):
    """Output model for the synthesis node."""
    answer: str = Field(
        description="A concise English answer (max 80 words) without in-text citations."
    )
    cited_ids: List[int] = Field(
        description="A list of citation IDs (e.g., [1, 3]) that were used to generate the answer, in the order they should appear at the end of the answer. These IDs correspond to the original numbering in the provided documents."
    )

# --- Node Implementations ---

def get_llm():
    """Initializes and returns the appropriate LLM based on available API keys."""
    # Load .env file from the project root
    load_dotenv(find_dotenv())
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if google_api_key:
        logger.info("--- Using Google Generative AI ---")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    elif openai_api_key:
        logger.info("--- Using OpenAI ---")
        return ChatOpenAI(temperature=0)
    else:
        raise ValueError("No LLM API key found. Please set either GOOGLE_API_KEY or OPENAI_API_KEY.")

# Initialize the LLM. We use a temperature of 0 for deterministic outputs.
llm = get_llm()

def log_error(step: str, error_type: str, message: str) -> dict:
    return {
        "errors": [{
            "step": step,
            "error_type": error_type,
            "message": message
        }]
    }

def generate_queries_node(state: GraphState) -> dict:
    """Generates initial search queries based on the user's question."""
    logger.info("--- Node: Generate Queries ---")
    question = state["question"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant. Your task is to generate a set of 3-5 diverse and relevant search queries based on a user's question. Return the queries as a JSON object."),
        ("user", f"User question: {question}")
    ])

    llm_with_tools: Runnable = llm.with_structured_output(GenerateQueriesOutput)
    chain = prompt | llm_with_tools

    try:
        result = chain.invoke({})
        return { "queries": result.queries }
    except RateLimitError as e:
        return { "queries": [], **log_error("generate", "RateLimit", str(e)) }
    except Exception as e:
        return { "queries": [], **log_error("generate", "LLMFailure", str(e)) }


def web_search_node(state: GraphState) -> dict:
    """Performs web searches for the given queries and updates the documents in the state."""
    logger.info("--- Node: Web Search ---")
    queries = state["queries"]
    
    search_tool = WebSearchTool()
    # Run the async tool in a sync context
    try:
        documents = asyncio.run(search_tool.run_concurrent(queries))
        
        # Append new documents to existing ones
        existing_docs = state.get("documents", [])
        all_docs = existing_docs + documents
        
        return {"documents": all_docs, "queries": []} # Clear queries after search
    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            return { "documents": state.get("documents", []), **log_error("search", "RateLimit", str(e)) }
        else:
            return { "documents": state.get("documents", []), **log_error("search", "HTTPError", str(e)) }

def reflect_node(state: GraphState) -> dict:
    """Reflects on the gathered information and decides if more searching is needed."""
    logger.info("--- Node: Reflect ---")
    question = state["question"]
    documents = state["documents"]

    # Format documents for the prompt
    doc_snippets = "\n".join([f"- {doc['content']}" for doc in documents])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research analyst. You need to decide if the current search results are sufficient to answer the user's question. If not, generate new queries. Your output must be a JSON object."),
        ("user", f"Original Question: {question}\n\nSearch Results:\n{doc_snippets}\n\nBased on these results, is there enough information to provide a comprehensive answer? If not, what new queries should be run?")
    ])
    
    llm_with_tools = llm.with_structured_output(ReflectOutput)
    chain = prompt | llm_with_tools
    
    result = chain.invoke({})
    
    logger.info(f"Reflection: Need more info? {result.need_more}")
    return {"need_more": result.need_more, "queries": result.new_queries or [], "loop_count": state.get("loop_count", 0) + 1}

def synthesize_node(state: GraphState) -> dict:
    """Synthesizes the final answer from the gathered documents."""
    logger.info("--- Node: Synthesize ---")
    question = state["question"]
    documents = state["documents"]

    if state.get("errors", []):
        return {"final_answer": "I'm unable to provide a complete answer at this time because of an error. Please try again later.", "citations": []}

    if not documents:
        return {"final_answer": "No information found.", "citations": []}

    # Prepare documents for the prompt with their original IDs
    doc_for_llm_prompt = []
    original_citations_map = {} # Map original ID to full document
    for i, doc in enumerate(documents):
        citation_id = i + 1
        doc_for_llm_prompt.append(
            f"[Citation {citation_id}] URL: {doc['url']}\nTitle: {doc['title']}\nContent: {doc['content']}"
        )
        original_citations_map[citation_id] = doc # Store the full document for later retrieval

    doc_for_llm_prompt_str = "\n".join(doc_for_llm_prompt)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a report writer. Your task is to synthesize a concise English answer (max 80 words) based on the provided documents. Do NOT include citations within the answer text. Instead, provide a list of the IDs of the citations you used to formulate the answer. These IDs correspond to the original numbering in the provided documents. Return a JSON object with the answer and the list of cited IDs."),
        ("user", f"Question: {question}\n\nDocuments:\n{doc_for_llm_prompt_str}")
    ])

    llm_with_tools = llm.with_structured_output(SynthesizeOutput)
    chain = prompt | llm_with_tools

    llm_result = chain.invoke({})

    final_answer_text = llm_result.answer
    cited_original_ids_from_llm = llm_result.cited_ids

    # Construct the final citation string [1][2]... and map original IDs to new sequential IDs
    citation_string_at_end = ""
    final_citations_list = []
    seen_urls = set() # To ensure unique citations in the final list
    original_to_new_id_map = {}

    for original_id in cited_original_ids_from_llm:
        doc = original_citations_map.get(original_id)
        if doc and doc['url'] not in seen_urls:
            new_id = len(final_citations_list) + 1
            final_citations_list.append({
                "id": new_id,
                "url": doc['url'],
                "title": doc['title']
            })
            seen_urls.add(doc['url'])
            original_to_new_id_map[original_id] = new_id
    
    # Use the mapping to create the citation string with new sequential IDs
    if cited_original_ids_from_llm:
        citation_string_at_end = "".join([f"[{original_to_new_id_map.get(_id, _id)}]" for _id in cited_original_ids_from_llm])

    # Combine answer and citations at the end
    full_answer = f"{final_answer_text}{citation_string_at_end}".strip()
    output = {"final_answer": full_answer, "citations": final_citations_list}
    logger.info(f"Final output: {output}")

    return output