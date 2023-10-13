from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from supervisely.app.singleton import Singleton


class WebsocketManager(metaclass=Singleton):
    def __init__(self, path="/sly-app-ws"):
        self.app = None
        self.path = path
        self.active_connections: List[WebSocket] = []

    def set_app(self, app: FastAPI):
        if self.app is not None:
            return
        self.app = app
        self.app.add_api_websocket_route(path=self.path, endpoint=self.endpoint)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, d: dict):
        # if self.app is None:
        #     raise ValueError(
        #         "WebSocket is not initialized, use Websocket middleware for that"
        #     )
        for connection in self.active_connections:
            await connection.send_json(d)

    async def endpoint(self, websocket: WebSocket):
        await self.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
        except WebSocketDisconnect:
            self.disconnect(websocket)
