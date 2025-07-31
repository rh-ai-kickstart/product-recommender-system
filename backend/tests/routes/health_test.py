import pytest
from httpx import ASGITransport, AsyncClient

from database.db import get_db
from main import app


class FakeFailureSession:
    async def execute(self, stmt):
        raise Exception("DB failure")


class FakeSuccessSession:
    async def execute(self, stmt):
        assert str(stmt) == "SELECT 1"

        class Result:
            def scalar(self):
                return 1

        return Result()


@pytest.fixture
def override_get_db():
    def _make_override(session_cls):
        async def fake_get_db():
            yield session_cls()

        app.dependency_overrides[get_db] = fake_get_db

    yield _make_override
    app.dependency_overrides.clear()


async def run_with_client(test_func):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        return await test_func(ac)


@pytest.mark.asyncio
async def test_liveness():
    async def inner(client):
        response = await client.get("/health/live")
        assert response.status_code == 200

    await run_with_client(inner)


@pytest.mark.asyncio
async def test_readiness_success(override_get_db):
    override_get_db(FakeSuccessSession)

    async def inner(client):
        response = await client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

    await run_with_client(inner)


@pytest.mark.asyncio
async def test_readiness_failure(override_get_db):
    override_get_db(FakeFailureSession)

    async def inner(client):
        response = await client.get("/health/ready")
        assert response.status_code == 503

    await run_with_client(inner)
