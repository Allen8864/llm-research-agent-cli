import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio
from src.agent.graph import app
from src.agent.state import GraphState
from src.agent.tools import WebSearchTool
import aiohttp


@pytest.fixture
def mock_search_results():
    """Fixture providing mock search results about Paris."""
    return [
        {
            "url": "https://example.com/article1",
            "title": "Test Article 1",
            "content": "The capital of France is Paris. It is known as the City of Light."
        },
        {
            "url": "https://example.com/article2", 
            "title": "Test Article 2",
            "content": "Paris has many famous landmarks including the Eiffel Tower and Louvre Museum."
        }
    ]


@pytest.fixture
def mock_search_tool(mock_search_results):
    """Fixture providing a mock WebSearchTool that returns predefined results."""
    mock_tool = WebSearchTool()
    mock_tool.run_concurrent = AsyncMock(return_value=mock_search_results)
    return mock_tool


@pytest.fixture
def empty_search_tool():
    """Fixture providing a mock WebSearchTool that returns empty results."""
    mock_instance = AsyncMock()
    mock_instance.run_concurrent.return_value = []
    return mock_instance


@pytest.fixture
def rate_limited_search_tool():
    """Fixture providing a mock WebSearchTool that raises a rate limit error."""
    mock_instance = AsyncMock()
    http_error = aiohttp.ClientResponseError(
        request_info=MagicMock(),
        history=(),
        status=429,
        message="Too Many Requests",
        headers={}
    )
    mock_instance.run_concurrent.side_effect = http_error
    return mock_instance


@pytest.fixture
def timeout_search_tool():
    """Fixture providing a mock WebSearchTool that simulates a timeout."""
    mock_instance = AsyncMock()
    timeout_error = asyncio.TimeoutError("Request timed out")
    mock_instance.run_concurrent.side_effect = timeout_error
    return mock_instance


@pytest.fixture
def two_round_search_tool():
    """Fixture providing a mock WebSearchTool that returns different results in two rounds."""
    mock_instance = AsyncMock()
    
    # First round: Partial information about World Cup
    first_round_results = [
        {
            "url": "https://example.com/worldcup1",
            "title": "FIFA World Cup 2022",
            "content": "The 2022 FIFA World Cup was held in Qatar from November to December. It was the first World Cup held in the Middle East."
        },
        {
            "url": "https://example.com/worldcup2",
            "title": "World Cup History",
            "content": "The FIFA World Cup is held every four years. The 2022 tournament featured 32 teams competing in eight groups."
        }
    ]
    
    # Second round: Information about the winner
    second_round_results = [
        {
            "url": "https://example.com/worldcup3",
            "title": "Argentina Wins World Cup",
            "content": "Argentina won the 2022 FIFA World Cup, defeating France in the final match on penalties after a 3-3 draw."
        },
        {
            "url": "https://example.com/worldcup4",
            "title": "Messi's Victory",
            "content": "Lionel Messi led Argentina to victory in the 2022 World Cup, finally winning the trophy that had eluded him throughout his career."
        }
    ]
    
    # Configure the mock to return different results on consecutive calls
    mock_instance.run_concurrent.side_effect = [first_round_results, second_round_results]
    return mock_instance


@pytest.fixture
def base_state():
    """Fixture providing a base GraphState for testing."""
    return GraphState(
        question="",
        queries=[],
        documents=[],
        need_more=False,
        final_answer="",
        citations=[],
        loop_count=0,
        max_iter=2,
        errors=[]
    )


@pytest.fixture
def france_question_state(base_state):
    """Fixture providing a GraphState with a question about France."""
    base_state["question"] = "What is the capital of France?"
    return base_state


@pytest.fixture
def worldcup_question_state(base_state):
    """Fixture providing a GraphState with a question about the World Cup."""
    base_state["question"] = "Who won the 2022 FIFA World Cup?"
    return base_state


@pytest.fixture
def atlantis_question_state(base_state):
    """Fixture providing a GraphState with a question about Atlantis and some queries."""
    base_state["question"] = "What is the population of fictional city Atlantis?"
    base_state["queries"] = ["atlantis population", "fictional city atlantis"]
    return base_state


@pytest.fixture
def france_query_state(france_question_state):
    """Fixture providing a GraphState with a query about France."""
    france_question_state["queries"] = ["capital of France"]
    return france_question_state


@pytest.mark.parametrize("state_fixture", ["france_question_state"])
def test_agent_happy_path(mock_search_tool, request, state_fixture):
    """Test the happy path where initial search results are sufficient."""
    # Get the parameterized state
    initial_state = request.getfixturevalue(state_fixture)
    
    # Patch the WebSearchTool class
    with patch('src.agent.nodes.WebSearchTool', return_value=mock_search_tool):
        # Run the agent
        final_state = app.invoke(initial_state)
        
        # Verify the final answer exists and contains citations
        assert final_state["final_answer"], "Final answer should not be empty"
        assert "[1]" in final_state["final_answer"] or "[2]" in final_state["final_answer"], "Answer should include citations"
        
        # Verify citations are present
        assert len(final_state["citations"]) > 0, "Citations should be present"
        assert all(isinstance(c, dict) for c in final_state["citations"]), "Citations should be dictionaries"
        assert all("url" in c and "title" in c for c in final_state["citations"]), "Citations should have url and title"
        
        # Verify need_more is false since initial results were sufficient
        assert not final_state["need_more"], "need_more should be False for sufficient results"
        
        # Verify the answer is relevant to the question
        assert "Paris" in final_state["final_answer"], "Answer should mention Paris"


def test_agent_no_results(empty_search_tool, atlantis_question_state):
    """Test the agent's handling of empty search results."""
    with patch('src.agent.nodes.WebSearchTool', return_value=empty_search_tool):
        final_state = app.invoke(atlantis_question_state)

        assert not final_state["need_more"]
        assert final_state["final_answer"]
        assert any(
            p in final_state["final_answer"].lower()
            for p in ["no information", "could not find", "unable to find",
                      "no data", "insufficient information", "not available"]
        )
        assert final_state["citations"] == []


def test_agent_rate_limit_error(rate_limited_search_tool, france_query_state):
    """Test the agent's handling of HTTP 429 rate limit errors."""
    with patch('src.agent.nodes.WebSearchTool', return_value=rate_limited_search_tool):
        final_state = app.invoke(france_query_state)
        
        # Verify the agent handles the rate limit error gracefully
        assert final_state["final_answer"], "Final answer should not be empty"
        assert any(
            p in final_state["final_answer"].lower()
            for p in ["rate limit", "too many requests", "try again later", 
                     "temporarily unavailable", "service unavailable"]
        ), "Answer should mention rate limiting"
        assert not final_state["need_more"], "need_more should be False after rate limit error"
        assert final_state["citations"] == [], "Citations should be empty for rate limit error"
        
        # Verify the search tool was called
        rate_limited_search_tool.run_concurrent.assert_called_once()


def test_agent_timeout_error(timeout_search_tool, france_query_state):
    """Test the agent's handling of timeout errors during web search."""
    with patch('src.agent.nodes.WebSearchTool', return_value=timeout_search_tool):
        final_state = app.invoke(france_query_state)
        
        # Verify the agent handles the timeout error gracefully
        assert final_state["final_answer"], "Final answer should not be empty"
        assert "error" in final_state["final_answer"].lower(), "Answer should mention an error occurred"
        assert not final_state["need_more"], "need_more should be False after timeout error"
        assert final_state["citations"] == [], "Citations should be empty for timeout error"
        
        # Verify the search tool was called
        timeout_search_tool.run_concurrent.assert_called_once()


def test_agent_two_round_search(two_round_search_tool, worldcup_question_state):
    """Test the agent performing a two-round search with reflection in between."""
    # Create a patched reflect_node that will set need_more=True on first call
    # and increment loop_count
    def patched_reflect_node(state):
        # Increment loop count
        current_loop_count = state.get("loop_count", 0)
        
        # First round: need more info
        if current_loop_count == 0:
            return {
                "need_more": True,
                "queries": ["Argentina World Cup 2022 winner"],
                "loop_count": current_loop_count + 1
            }
        # Second round: we have enough info
        else:
            return {
                "need_more": False,
                "queries": [],
                "loop_count": current_loop_count + 1
            }
    
    with patch('src.agent.nodes.WebSearchTool', return_value=two_round_search_tool), \
         patch('src.agent.nodes.reflect_node', side_effect=patched_reflect_node):
        
        # Run the agent
        final_state = app.invoke(worldcup_question_state)
        
        # Verify the search tool was called twice
        assert two_round_search_tool.run_concurrent.call_count == 2, "Search tool should be called twice"
        
        # Verify the final answer exists and contains citations
        assert final_state["final_answer"], "Final answer should not be empty"
        assert "[1]" in final_state["final_answer"] or "[2]" in final_state["final_answer"], "Answer should include citations"
        
        # Verify citations are present
        assert len(final_state["citations"]) > 0, "Citations should be present"
        
        # Verify the answer is correct and comprehensive
        assert "Argentina" in final_state["final_answer"], "Answer should mention Argentina as the winner"
        assert "France" in final_state["final_answer"], "Answer should mention France as the finalist"
        
        # Verify that documents from at least one search round were used
        urls = [citation["url"] for citation in final_state["citations"]]
        # The agent might prioritize the most relevant information from the second round
        assert any("worldcup" in url for url in urls), "Should include citations from search results"
        
        # Verify the loop_count was incremented
        assert final_state["loop_count"] > 0, "Loop count should be incremented"

