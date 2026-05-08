from __future__ import annotations

import asyncio
import logging
import os
import signal

import grpc
from dotenv import load_dotenv
from grpc_reflection.v1alpha import reflection

from llm.v1 import llm_pb2, llm_pb2_grpc
from openai_client import OpenAIStreamingClient
from service import LLMGatewayServicer


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc


async def _serve() -> None:
    load_dotenv()
    log_level = logging.DEBUG if os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG" else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger("rag-llm-gateway")

    port = _env_int("GRPC_PORT", 50051)

    openai_client = OpenAIStreamingClient()

    server = grpc.aio.server()
    llm_pb2_grpc.add_LLMGatewayServicer_to_server(
        LLMGatewayServicer(openai_client), server
    )

    service_names = (
        llm_pb2.DESCRIPTOR.services_by_name["LLMGateway"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    log.info("Rag-llm-gateway listening on :%d", port)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_signal(signum: int) -> None:
        log.info("Received signal %s, shutting down", signum)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal, sig)
        except NotImplementedError:
            # Windows: signal handlers via add_signal_handler are unsupported.
            signal.signal(sig, lambda s, _f: _on_signal(s))

    await stop_event.wait()
    await server.stop(grace=5)
    await openai_client.close()


def serve() -> None:
    asyncio.run(_serve())


if __name__ == "__main__":
    serve()
