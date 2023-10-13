import os
import typing

from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException, status
from starlette.datastructures import Headers
from starlette.responses import FileResponse, Response, StreamingResponse
from starlette.staticfiles import NotModifiedResponse
from starlette.types import Scope
from supervisely.video.video import ALLOWED_VIDEO_EXTENSIONS

PathLike = typing.Union[str, "os.PathLike[str]"]


class CustomStaticFiles(StaticFiles):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def file_response(
        self,
        full_path: PathLike,
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        method = scope["method"]
        request_headers = Headers(scope=scope)
        range_header = request_headers.get("Range")

        def _send_bytes_range_requests(
            file_obj: typing.BinaryIO, start: int, end: int, chunk_size: int = 10_000
        ):
            with file_obj as f:
                f.seek(start)
                pos = f.tell()
                while pos <= end:
                    read_size = min(chunk_size, end + 1 - pos)
                    pos = f.tell()
                    yield f.read(read_size)

        def _get_range_header(range_header: str, file_size: int) -> typing.Tuple[int, int]:
            def _invalid_range():
                return HTTPException(
                    status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                    detail=f"Invalid request range (Range:{range_header!r})",
                )

            try:
                h = range_header.replace("bytes=", "").split("-")
                start = int(h[0]) if h[0] != "" else 0
                end = int(h[1]) if h[1] != "" else file_size - 1
            except ValueError:
                raise _invalid_range()

            if start > end or start < 0 or end > file_size - 1:
                raise _invalid_range()
            return start, end

        if range_header is not None and Path(full_path).suffix in ALLOWED_VIDEO_EXTENSIONS:
            file_size = stat_result.st_size

            headers = {
                "content-type": "video/mp4",
                "accept-ranges": "bytes",
                "content-encoding": "identity",
                "content-length": str(file_size),
                "access-control-expose-headers": (
                    "content-type, accept-ranges, content-length, "
                    "content-range, content-encoding"
                ),
            }

            start, end = _get_range_header(range_header, file_size)
            size = end - start + 1
            headers["content-length"] = str(size)
            headers["content-range"] = f"bytes {start}-{end}/{file_size}"
            status_code = status.HTTP_206_PARTIAL_CONTENT

            response = StreamingResponse(
                _send_bytes_range_requests(open(full_path, mode="rb"), start, end),
                headers=headers,
                status_code=status_code,
            )

        else:
            response = FileResponse(
                full_path, status_code=status_code, stat_result=stat_result, method=method
            )
        if self.is_not_modified(response.headers, request_headers):
            return NotModifiedResponse(response.headers)
        return response
