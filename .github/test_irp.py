import re
import tomllib
import pytest
import pathlib

EIGHT_MB = 8_388_608  # in bytes

deliverables_dir = pathlib.Path("./deliverables")


@pytest.fixture(scope="session")
def username():
    """Extract username from the repo name."""
    repo_name = pathlib.Path(__file__).parents[1].name
    return re.search(r'-(.*)', repo_name).group(1)


@pytest.fixture(scope="session")
def title():
    """Extract title from title.toml."""
    with open(pathlib.Path("./title/title.toml"), "r", encoding="utf-8") as file:
        yield tomllib.loads(file.read())['title']


@pytest.fixture(scope="session")
def project_plan(username):
    """Derive project plan path."""
    return deliverables_dir / f"{username}-project-plan.pdf"


@pytest.fixture(scope="session")
def final_report(username):
    """Derive final report path."""
    return deliverables_dir / f"{username}-final-report.pdf"


@pytest.fixture(scope="session")
def presentation(username):
    """Derive presentation path."""
    return deliverables_dir / f"{username}-presentation.pdf"


class TestTitle:
    def test_title(self, title):
        assert isinstance(title, str)
        assert title


class TestProjectPlan:
    def test_name(self, project_plan):
        assert project_plan.is_file()

    def test_size(self, project_plan):
        assert project_plan.stat().st_size <= EIGHT_MB


class TestFinalReport:
    def test_name(self, final_report):
        assert final_report.is_file()

    def test_size(self, final_report):
        assert final_report.stat().st_size <= EIGHT_MB


class TestPresentation:
    def test_name(self, presentation):
        assert presentation.is_file()

    def test_size(self, presentation):
        assert presentation.stat().st_size <= EIGHT_MB
