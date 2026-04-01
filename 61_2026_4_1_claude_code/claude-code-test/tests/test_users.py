import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db

TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_database():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture()
def client():
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_create_user_success(client):
    r = client.post("/users", json={"name": "Alice", "email": "alice@example.com"})
    assert r.status_code == 201
    d = r.json()
    assert d["name"] == "Alice"
    assert d["email"] == "alice@example.com"
    assert "id" in d
    assert "created_at" in d


def test_create_user_duplicate_email(client):
    payload = {"name": "Bob", "email": "bob@example.com"}
    client.post("/users", json=payload)
    r = client.post("/users", json=payload)
    assert r.status_code == 409


def test_create_user_invalid_email(client):
    r = client.post("/users", json={"name": "Charlie", "email": "not-an-email"})
    assert r.status_code == 422


def test_create_user_missing_name(client):
    r = client.post("/users", json={"email": "dave@example.com"})
    assert r.status_code == 422


def test_create_user_missing_email(client):
    r = client.post("/users", json={"name": "Eve"})
    assert r.status_code == 422
