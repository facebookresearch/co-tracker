# # -*- coding: utf-8 -*-
# """
#
# requests_toolbelt.multipart.decoder
# ===================================
#
# This holds all the implementation details of the MultipartDecoder. https://github.com/requests/toolbelt/pull/222
#
# """
#
#
# import sys
# import email.parser
# from requests_toolbelt.multipart.encoder import encode_with
# from requests.structures import CaseInsensitiveDict
#
#
# def _split_on_find(content, bound):
#     point = content.find(bound)
#     return content[:point], content[point + len(bound):]
#
#
# class ImproperBodyPartContentException(Exception):
#     pass
#
#
# class NonMultipartContentTypeException(Exception):
#     pass
#
#
# def _header_parser(string, encoding):
#     major = sys.version_info[0]
#     if major == 3:
#         string = string.decode(encoding)
#     headers = email.parser.HeaderParser().parsestr(string).items()
#     return (
#         (encode_with(k, encoding), encode_with(v, encoding))
#         for k, v in headers
#     )
#
#
# class BodyPart(object):
#     """
#
#     The ``BodyPart`` object is a ``Response``-like interface to an individual
#     subpart of a multipart response. It is expected that these will
#     generally be created by objects of the ``MultipartDecoder`` class.
#
#     Like ``Response``, there is a ``CaseInsensitiveDict`` object named headers,
#     ``content`` to access bytes, ``text`` to access unicode, and ``encoding``
#     to access the unicode codec.
#
#     """
#
#     def __init__(self, content, encoding):
#         self.encoding = encoding
#         headers = {}
#         # Split into header section (if any) and the content
#         if b'\r\n\r\n' in content:
#             first, self.content = _split_on_find(content, b'\r\n\r\n')
#             if first != b'':
#                 headers = _header_parser(first.lstrip(), encoding)
#         else:
#             raise ImproperBodyPartContentException(
#                 'content does not contain CR-LF-CR-LF'
#             )
#         self.headers = CaseInsensitiveDict(headers)
#
#     @property
#     def text(self):
#         """Content of the ``BodyPart`` in unicode."""
#         return self.content.decode(self.encoding)
#
#
# class MultipartDecoder(object):
#     """
#
#     The ``MultipartDecoder`` object parses the multipart payload of
#     a bytestring into a tuple of ``Response``-like ``BodyPart`` objects.
#
#     The basic usage is::
#
#         import requests
#         from requests_toolbelt import MultipartDecoder
#
#         response = request.get(url)
#         decoder = MultipartDecoder.from_response(response)
#         for part in decoder.parts:
#             print(part.headers['content-type'])
#
#     If the multipart content is not from a response, basic usage is::
#
#         from requests_toolbelt import MultipartDecoder
#
#         decoder = MultipartDecoder(content, content_type)
#         for part in decoder.parts:
#             print(part.headers['content-type'])
#
#     For both these usages, there is an optional ``encoding`` parameter. This is
#     a string, which is the name of the unicode codec to use (default is
#     ``'utf-8'``).
#
#     """
#     def __init__(self, content, content_type, encoding='utf-8'):
#         #: Original Content-Type header
#         self.content_type = content_type
#         #: Response body encoding
#         self.encoding = encoding
#         #: Parsed parts of the multipart response body
#         self.parts = tuple()
#         self.boundary = MultipartDecoder._find_boundary(content_type, encoding)
#         self._parse_body(content)
#
#     @staticmethod
#     def _find_boundary(content_type, encoding):
#         ct_info = tuple(x.strip() for x in content_type.split(';'))
#         mimetype = ct_info[0]
#         if mimetype.split('/')[0].lower() != 'multipart':
#             raise NonMultipartContentTypeException(
#                 "Unexpected mimetype in content-type: '{0}'".format(mimetype)
#             )
#         for item in ct_info[1:]:
#             attr, value = _split_on_find(
#                 item,
#                 '='
#             )
#             if attr.lower() == 'boundary':
#                 boundary = encode_with(value.strip('"'), encoding)
#         return boundary
#
#     @staticmethod
#     def _fix_first_part(part, boundary_marker):
#         bm_len = len(boundary_marker)
#         if boundary_marker == part[:bm_len]:
#             return part[bm_len:]
#         else:
#             return part
#
#     def _parse_body(self, content):
#         boundary = b''.join((b'--', self.boundary))
#
#         def body_part(part):
#             fixed = MultipartDecoder._fix_first_part(part, boundary)
#             return BodyPart(fixed, self.encoding)
#
#         def test_part(part):
#             return (part != b'' and
#                     part != b'\r\n' and
#                     part[:4] != b'--\r\n' and
#                     part != b'--')
#
#         parts = content.split(b''.join((b'\r\n', boundary)))
#         self.parts = tuple(body_part(x) for x in parts if test_part(x))
#
#     @classmethod
#     def from_response(cls, response, encoding='utf-8'):
#         content = response.content
#         content_type = response.headers.get('content-type', None)
#         return cls(content, content_type, encoding)
#
#
# # This is thrown when the object is being reiterated from another place
# class AlreadyIteratedException(Exception):
#     pass
#
#
# # This is thrown when trying to skip to the next part without finishing to
# # stream the previous one
# class PreviousPartNotFinishedException(Exception):
#     pass
#
#
# class StreamPart(object):
#     def __init__(self, headers, encoding, iterator):
#         self.headers = headers
#         self.encoding = encoding
#         self._iterator = iterator
#         self._started = False
#         self._consumed = False
#         self._content = None
#         self._finished = False
#
#     def __iter__(self):
#         if self._started:
#             raise AlreadyIteratedException()
#         self._started = True
#         for typ, data in self._iterator():
#             if typ == 'done' and data is False:
#                 self._finished = True
#                 break
#             elif typ == 'stream':
#                 yield data
#             else:
#                 raise ImproperBodyPartContentException()
#
#     @property
#     def content(self):
#         if self._consumed:
#             return self._content
#         if self._started:
#             raise AlreadyIteratedException()
#         self._content = b''.join(self)
#         self._consumed = True
#         return self._content
#
#     @property
#     def text(self):
#         return self.content.decode(self.encoding)
#
#
# class MultipartStreamDecoder(object):
#     @classmethod
#     def from_response(cls, response, encoding='utf-8', chunk_size=10 * 1024,
#                       header_size_limit=None):
#         def content():
#             return response.raw.read(chunk_size)
#         content_type = response.headers.get('content-type', None)
#         return cls(content, content_type, encoding, header_size_limit)
#
#     def __init__(self, stream_read_func, content_type, encoding='utf-8',
#                  header_size_limit=None):
#         self.content_type = content_type
#         self.encoding = encoding
#         self._stream_read_func = stream_read_func
#         self._header_size_limit = header_size_limit
#         self._boundary = MultipartDecoder._find_boundary(content_type,
#                                                          encoding)
#         self._splitter = StreamSplitter()
#         self._boundary = b''.join((b'--', self._boundary))
#         self._boundary_split = b''.join((b'\r\n', self._boundary))
#         self._state = 0
#         self._found = False
#         self._started = False
#         self._finished = False
#
#     # Call this to drain stream when error occured, or you decide
#     # not to read all data
#     def close(self):
#         if not self._finished:
#             while True:
#                 try:
#                     data = self._stream_read_func()
#                     if not data:
#                         break
#                 # Protection if _stream_read_func is an generator
#                 except StopIteration:
#                     break
#                 finally:
#                     self._finished = True
#
#     # The instance can be used as an context manager for automatic
#     # draining the stream for re-use
#     def __enter__(self):
#         return self
#
#     def __exit__(self, exc_type, exc_value, exc_traceback):
#         self.close()
#
#     def __iter__(self):
#         if self._started:
#             raise AlreadyIteratedException()
#         self._started = True
#         # This is for guarding against iterating before finishing to
#         # iterate on the current part.
#         _current_stream = None
#         for typ, data in self._iterator():
#             if _current_stream and not _current_stream._finished:
#                 raise PreviousPartNotFinishedException()
#             if typ == 'headers':
#                 _current_stream = StreamPart(
#                     data, self.encoding, self._iterator
#                 )
#                 yield _current_stream
#             else:
#                 raise ImproperBodyPartContentException()
#
#     def _iterator(self):
#         while True:
#             data = self._stream_read_func()
#             # This persumes that if data returned empty once it won't return
#             # anything again (EOS)
#             if not self._found and not data:
#                 self._finished = True
#                 break
#             # This part mimics the _fix_first_part logic from above
#             if self._state == 0:
#                 should_be_empty_or_crlf, self._found = self._splitter.stream(
#                     data, self._boundary, True
#                 )
#                 if should_be_empty_or_crlf:
#                     if should_be_empty_or_crlf != b'\r\n':
#                         raise ImproperBodyPartContentException()
#                     self._state = 1
#                     continue
#                 if self._found:
#                     self._state = 1
#                     continue
#             # Parse the headers
#             elif self._state == 1:
#                 headers, self._found = self._splitter.stream(data, b'\r\n\r\n',
#                                                              True)
#                 if headers:
#                     headers = _header_parser(headers.lstrip(), self.encoding)
#                     headers = CaseInsensitiveDict(headers)
#                     self._state = 2
#                     yield 'headers', headers
#                     continue
#                 # No headers found
#                 if self._found:
#                     headers = CaseInsensitiveDict({})
#                     self._state = 2
#                     yield 'headers', headers
#                     continue
#                 # This is to protect against malformed input where a header
#                 # does not exist in a limit for performence reasons
#                 if self._header_size_limit:
#                     if (self._splitter.leftover_length >
#                             self._header_size_limit):
#                         raise ImproperBodyPartContentException()
#             # Stream the part
#             elif self._state == 2:
#                 stream, self._found = self._splitter.stream(
#                     data, self._boundary_split
#                 )
#                 if stream:
#                     yield 'stream', stream
#                 # boundary_split found, end of part
#                 if self._found:
#                     self._state = 1
#                     yield 'done', False
#                     continue
#
#
# class StreamSplitter(object):
#     def __init__(self):
#         self.leftover = b''
#
#     def stream(self, data, split_data, return_only_full=False):
#         self.leftover += data
#         index = self.leftover.find(split_data)
#         if return_only_full:
#             if index > -1:
#                 ret = self.leftover[:index]
#                 self.leftover = self.leftover[index + len(split_data):]
#                 found = True
#             else:
#                 ret = b''
#                 found = False
#         else:
#             if index > -1:
#                 ret = self.leftover[:index]
#                 self.leftover = self.leftover[index + len(split_data):]
#                 found = True
#             elif len(self.leftover) >= len(split_data):
#                 ret = self.leftover[:-len(split_data)]
#                 self.leftover = self.leftover[-len(split_data):]
#                 found = False
#             else:
#                 ret = b''
#                 found = False
#         return ret, found
#
#     @property
#     def leftover_length(self):
#         return len(self.leftover)



# remote storage download api
# MultipartStreamDecoder has bug
#     def _download_batch(self, paths, is_stream=True):
#         def parse_name(headers):
#             name = part.headers[b'Content-Disposition'].decode("utf-8").split('name="')[1].split('"')[0]
#             return name
#         response = self._api.post('remote-storage.bulk.download', paths, stream=is_stream)
#         with MultipartStreamDecoder.from_response(response, chunk_size=1024 * 1024) as decoder:
#             for part in decoder:
#                 #print(part.headers)
#                 remote_path = parse_name(part.headers)
#                 for stream in part:
#                     yield remote_path, stream
#                 # print(part.content) # Read comment below
#
#     def download_paths(self, remote_paths, local_paths):
#         path_map = {remote: local for remote, local in zip(remote_paths, local_paths)}
#         last_file = ""
#         w = None
#         for current_path, resp_part in self._download_batch(remote_paths, is_stream=True):
#             if last_file != current_path:
#                 if w is not None:
#                     w.close()
#                 w = open(path_map[current_path], 'wb')
#                 last_file = current_path
#             w.write(resp_part)
#         if w is not None:
#             w.close()