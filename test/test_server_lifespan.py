from fastapi import FastAPI

import server


class DummyManager:
    def __init__(self):
        self.shutdown_called = False

    def shutdown(self):
        self.shutdown_called = True


def test_server_lifespan_sets_and_clears_manager(monkeypatch):
    dummy = DummyManager()
    monkeypatch.setattr(server, "create_model_manager_from_env", lambda: dummy)

    async def run_lifespan():
        async with server.lifespan(server.app):
            assert server.model_manager is dummy
        assert server.model_manager is None
        assert dummy.shutdown_called

    import asyncio
    asyncio.run(run_lifespan())
