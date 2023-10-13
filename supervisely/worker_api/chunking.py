# coding: utf-8

import math
import supervisely as sly


class ChunkSplitter:
    def __init__(self, tot_size, chunk_size):
        self.tot_size = tot_size
        self.chunk_size = chunk_size

    def __next__(self):
        for curr_pos in range(0, self.tot_size, self.chunk_size):
            curr_chunk_size = min(self.tot_size - curr_pos, self.chunk_size)
            yield (curr_pos, curr_chunk_size)

    def __iter__(self):
        return next(self)

    @property
    def chunk_cnt(self):
        return math.ceil(self.tot_size / self.chunk_size)


class ChunkedFileWriter:
    def __init__(self, file_path):
        self.handler = None
        self.path = file_path

        self.total_size = 0
        self.written_bytes = 0

    @property
    def file_path(self):
        return self.path

    # with proto chunks
    def write(self, chunk):
        if chunk.total_size != 0:
            self.total_size = chunk.total_size
        if self.written_bytes == 0:
            sly.fs.ensure_base_path(self.path)
            self.handler = open(self.path, 'wb')

        self.handler.write(chunk.buffer)
        self.written_bytes += len(chunk.buffer)

    # remove file on error
    def close_and_check(self):
        if (self.handler is not None) and (not self.handler.closed):
            self.handler.close()
        if (self.total_size != 0) and (self.written_bytes != self.total_size):
            sly.fs.silent_remove(self.path)
            return False
        return True


class ChunkedFileReader:
    def __init__(self, fpath, chunk_size):
        self.fpath = fpath
        self.file_size = sly.fs.get_file_size(fpath)  # bytes
        self.splitter = ChunkSplitter(self.file_size, chunk_size)

    def __next__(self):
        with open(self.fpath, 'rb') as file_:
            for chunk_start, chunk_size in self.splitter:
                chunk_bytes = file_.read(chunk_size)
                yield chunk_bytes

    def __iter__(self):
        return next(self)


def load_to_memory_chunked(iterable_resp):
    total_size = None
    b_data = bytearray()
    for chunk in iterable_resp:
        b_data.extend(chunk.buffer)
        if chunk.total_size != 0:
            total_size = chunk.total_size  # last non-zero value

    b_data_size = len(b_data)
    if (total_size is not None) and (total_size != b_data_size):
        raise RuntimeError('Incomplete input stream (by total_size). Expected: {}, Received: {}.'
                           .format(total_size, b_data_size))
    return b_data


def load_to_memory_chunked_image(iterable_resp):
    def iter_chunk_image(r):
        for chunk_image in r:
            yield chunk_image.chunk

    res = load_to_memory_chunked(iter_chunk_image(iterable_resp))
    return res
