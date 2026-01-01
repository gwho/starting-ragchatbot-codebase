"""
Pytest configuration and fixtures for RAG system tests
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add backend directory to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from vector_store import SearchResults
from models import Course, Lesson


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore with common test methods"""
    store = Mock()

    # Default successful search results
    store.search.return_value = SearchResults(
        documents=["Test content from lesson 1"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0}],
        distances=[0.5]
    )

    # Default link retrieval
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    store.get_course_link.return_value = "https://example.com/course"

    # Default course outline
    store.get_course_outline.return_value = {
        "title": "Test Course",
        "instructor": "Test Instructor",
        "course_link": "https://example.com/course",
        "lesson_count": 3,
        "lessons": [
            {"lesson_number": 0, "lesson_title": "Introduction"},
            {"lesson_number": 1, "lesson_title": "Getting Started"},
            {"lesson_number": 2, "lesson_title": "Advanced Topics"}
        ]
    }

    return store


@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager with tool execution"""
    manager = Mock()
    manager.execute_tool.return_value = "[Test Course - Lesson 1]\nTest content"
    manager.get_last_sources.return_value = [{
        "text": "Test Course - Lesson 1",
        "link": "https://example.com/lesson/1",
        "type": "lesson"
    }]
    manager.reset_sources.return_value = None
    return manager


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic API client"""
    client = Mock()

    # Mock successful response without tool use
    response = Mock()
    response.content = [Mock(type="text", text="Test response")]
    response.stop_reason = "end_turn"
    client.messages.create.return_value = response

    return client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI/OpenRouter API client"""
    client = Mock()

    # Mock successful response
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "Test response"
    response.choices[0].message.tool_calls = None
    client.chat.completions.create.return_value = response

    return client


@pytest.fixture
def sample_course():
    """Sample course object for testing"""
    return Course(
        title="Introduction to Testing",
        course_link="https://example.com/testing-course",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=0, title="Course Overview", lesson_link="https://example.com/lesson/0"),
            Lesson(lesson_number=1, title="Getting Started", lesson_link="https://example.com/lesson/1"),
            Lesson(lesson_number=2, title="Advanced Topics", lesson_link="https://example.com/lesson/2"),
        ]
    )


@pytest.fixture
def empty_search_results():
    """Empty search results for testing edge cases"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def search_results_with_error():
    """Search results with error for testing error handling"""
    return SearchResults.empty("Search failed: Connection timeout")
