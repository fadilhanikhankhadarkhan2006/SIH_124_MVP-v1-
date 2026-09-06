"""
Urban AI Fleet Intelligence — Monolithic Core Server
====================================================
Production-grade Hexagonal Architecture matching the Master Plan:
  - Inbound Port: MQTT Protobuf Ingress Adapter (fleet/+/telemetry)
  - Core: Concurrent KD-Tree 3-Meter Spatial Deduplication Engine
  - H3 Spatial Grid aggregation for municipal heatmaps
  - Outbound Port: WebSocket Broadcast Adapter to Dashboard
  - Outbound Port: REST API for vehicles, defects, H3 hex grid, and health
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import SERVER_HOST, SERVER_PORT, WS_ENDPOINT
from adapters.mqtt_adapter import MQTTInboundAdapter
from adapters.websocket_adapter import WebSocketBroadcastAdapter
from domain.deduplication import SpatialDeduplicationEngine
from core.ingestion_service import TelemetryIngestionService
from plugins.mape_plugin import ServerMAPEPlugin
from plugins.km_plugin import KnowledgeMediaPlugin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("Server")

# Hexagonal Adapters & Core
ws_adapter = WebSocketBroadcastAdapter()
dedup_engine = SpatialDeduplicationEngine()
km_plugin = KnowledgeMediaPlugin()
mape_plugin = ServerMAPEPlugin()

ingestion_service = TelemetryIngestionService(
    dedup_engine=dedup_engine,
    ws_broadcast=ws_adapter.broadcast,
    km_plugin=km_plugin,
)

server_loop: asyncio.AbstractEventLoop = None
latest_video_frame: bytes | None = None
video_frame_condition = asyncio.Condition()


def on_telemetry_received(reading):
    ingestion_service.handle_telemetry_sync(reading, loop=server_loop)


def on_heartbeat_received(hb):
    ingestion_service.handle_heartbeat(hb)


mqtt_adapter = MQTTInboundAdapter(
    on_telemetry=on_telemetry_received,
    on_heartbeat=on_heartbeat_received,
)
km_plugin.set_command_sender(mqtt_adapter.publish_command)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global server_loop
    server_loop = asyncio.get_running_loop()

    logger.info("=" * 60)
    logger.info("  🚀 Urban AI Fleet Intelligence Server — Starting Up")
    logger.info("=" * 60)

    mqtt_adapter.start()
    yield

    logger.info("[Server] Gracefully shutting down...")
    mqtt_adapter.stop()
    logger.info("[Server] Shutdown complete.")


app = FastAPI(
    title="Urban AI Fleet Intelligence",
    description="Edge-AI powered municipal road infrastructure monitoring platform",
    version="2.0.0-masterplan",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ],
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ─────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "online",
        "service": "Urban AI Fleet Intelligence",
        "version": "2.0.0-masterplan",
    }


@app.get("/health", tags=["Health"])
@app.get("/api/health", tags=["Health"])
async def health():
    return {
        "mqtt_connected": mqtt_adapter._connected,
        "ws_clients": ws_adapter.get_connection_count(),
        **ingestion_service.get_stats(),
    }


@app.get("/defects", tags=["Data"])
@app.get("/api/defects", tags=["Data"])
async def get_defects():
    markers = ingestion_service.get_all_defects()
    return {
        "count": len(markers),
        "defects": [asdict(m) for m in markers],
    }


@app.get("/vehicles", tags=["Data"])
@app.get("/api/vehicles", tags=["Data"])
async def get_vehicles():
    vehicles = ingestion_service.get_all_vehicles()
    return {
        "count": len(vehicles),
        "vehicles": [asdict(v) for v in vehicles],
    }


@app.get("/h3-grid", tags=["Data"])
@app.get("/api/h3-grid", tags=["Data"])
async def get_h3_grid():
    cells = ingestion_service.get_h3_grid()
    return {
        "count": len(cells),
        "cells": cells,
    }


@app.get("/stats", tags=["Data"])
@app.get("/api/stats", tags=["Data"])
async def get_stats():
    return ingestion_service.get_stats()


@app.post("/video/frame", tags=["Video"])
@app.post("/api/video/frame", tags=["Video"])
async def receive_video_frame(request: Request):
    """Accept one JPEG-encoded annotated frame from an edge node."""
    global latest_video_frame
    frame = await request.body()
    if not frame:
        return {"accepted": False, "reason": "empty frame"}

    async with video_frame_condition:
        latest_video_frame = frame
        video_frame_condition.notify_all()
    return {"accepted": True, "bytes": len(frame)}


async def video_mjpeg_stream():
    """Yield the latest annotated frame as a browser-compatible MJPEG stream."""
    while True:
        async with video_frame_condition:
            try:
                await asyncio.wait_for(video_frame_condition.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            frame = latest_video_frame

        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"


@app.get("/video/stream", tags=["Video"])
@app.get("/api/video/stream", tags=["Video"])
async def video_stream():
    return StreamingResponse(
        video_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── WebSocket Endpoint ─────────────────────────────

@app.websocket(WS_ENDPOINT)
async def websocket_dashboard(websocket: WebSocket):
    await ws_adapter.handle_connection(websocket)
