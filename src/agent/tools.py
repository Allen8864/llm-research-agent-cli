import os
import httpx
import asyncio
from typing import List, Dict, Any

from langchain_tavily import TavilySearch

# A simple mock tool for offline development and testing
async def mock_web_search(query: str) -> List[Dict[str, str]]:
    """A mock web search function that returns dummy results."""
    print(f"--- Mock Searching for: {query} ---")
    await asyncio.sleep(1) # Simulate network latency
    return []

async def tavily_web_search(query: str) -> List[Dict[str, str]]:
    """Performs a web search using the Tavily Search API."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("TAVILY_API_KEY is not set. Falling back to mock search.")
        return []

    # TavilySearchResults automatically picks up TAVILY_API_KEY from environment
    search = TavilySearch(max_results=3)
    
    try:
        # TavilySearchResults.arun returns a dictionary with a 'results' key
        raw_results = await search.arun(query)
        
        # Extract the list of results from the 'results' key
        results_list = raw_results.get('results', [])

        # Parse the results into the desired format
        parsed_results = []
        for res in results_list:
            # Tavily results are typically a list of dicts, each with 'url', 'title', 'content' or 'snippet'
            parsed_results.append({
                "url": res.get("url", ""),
                "title": res.get("title", ""),
                "content": res.get("content", res.get("snippet", ""))
            })
        return parsed_results
    except Exception as e:
        print(f"An error occurred during Tavily search: {e}")
        return []

class WebSearchTool:
    """
    A tool to perform concurrent web searches using either Tavily or a mock tool.
    It automatically de-duplicates results based on the 'url'.
    """
    async def run_concurrent(self, queries: List[str]) -> List[Dict[str, str]]:
        """
        Runs web searches for a list of queries concurrently.
        Prioritizes Tavily, then falls back to the mock search.
        """
        search_func = None
        if os.getenv("TAVILY_API_KEY"):
            search_func = tavily_web_search
            print("--- Using Tavily Search ---")
        else:
            search_func = mock_web_search
            print("--- Using Mock Search ---")
        
        tasks = [search_func(q) for q in queries]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten the list of lists and filter out any exceptions
        flat_results = []
        for res in results_lists:
            if isinstance(res, list):
                flat_results.extend(res)
        
        # De-duplicate results based on the 'url'
        seen_urls = set()
        unique_results = []
        for result in flat_results:
            if result['url'] not in seen_urls:
                unique_results.append(result)
                seen_urls.add(result['url'])
                
        return unique_results