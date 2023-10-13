# coding: utf-8

import traceback
import requests
import time

CONNECTION_ERROR = 'Temporary connection error, please wait ...'
AGENT_CONNECTION_ERROR = 'Temporary connection error (agent ping), please wait ...'

REQUEST_FAILED = 'Request has failed. This may be due to connection problems or invalid requests. '
SPECIAL_RECONNECT_ERROR = "Agent should call AgentConnected or AgentPing before attempting any other request"
RETRY_STATUS_CODES = {
    408,  # Request Timeout
    429,  # Too Many Requests
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
    509,  # Bandwidth Limit Exceeded (Apache)
    598,  # Network read timeout error
    599   # Network connect timeout error
}


def process_requests_exception(external_logger, exc, api_method_name, url,
                               verbose=True, swallow_exc=False, sleep_sec=None, response=None, retry_info=None):
    is_connection_error = isinstance(exc, (requests.exceptions.ConnectionError,
                                           requests.exceptions.Timeout,
                                           requests.exceptions.TooManyRedirects,
                                           requests.exceptions.ChunkedEncodingError))

    is_server_retryable_error = isinstance(exc, requests.exceptions.HTTPError) and \
                                hasattr(exc, 'response') and \
                                (exc.response.status_code in RETRY_STATUS_CODES)

    is_need_ping_error = False
    if isinstance(exc, requests.exceptions.HTTPError) and hasattr(exc, 'response') and (exc.response.status_code == 400):
        try:
            server_explanation = exc.response.json()
            is_need_ping_error = (server_explanation.get('error', None) == SPECIAL_RECONNECT_ERROR)
        except (AttributeError, ValueError):
            pass

    if is_connection_error or is_server_retryable_error:
        process_retryable_request(external_logger, exc, api_method_name, url,
                                  CONNECTION_ERROR,
                                  verbose=verbose, swallow_exc=swallow_exc, sleep_sec=sleep_sec, retry_info=retry_info)
    elif is_need_ping_error:
        process_retryable_request(external_logger, exc, api_method_name, url,
                                  AGENT_CONNECTION_ERROR,
                                  verbose=verbose, swallow_exc=swallow_exc, sleep_sec=sleep_sec, retry_info=retry_info)
    elif isinstance(exc, requests.exceptions.HTTPError):
        process_invalid_request(external_logger, exc, response, verbose)
    else:
        process_unhandled_request(external_logger, exc)


def process_retryable_request(external_logger, exc, api_method_name, url, user_message,
                              verbose=True, swallow_exc=False, sleep_sec=None, retry_info=None):
    if retry_info is not None:
        retry_idx = retry_info["retry_idx"]
        retry_limit = retry_info["retry_limit"]
        user_message = "{}:  Retrying ({}/{}).".format(user_message, retry_idx, retry_limit)
    if verbose:
        external_logger.warn(user_message, extra={'method': api_method_name, 'url': url, 'details': str(exc)})

    if sleep_sec is not None:
        time.sleep(sleep_sec)

    if not swallow_exc:
        raise exc


def process_invalid_request(external_logger, exc, response, verbose=True):
    if verbose:
        external_logger.warn(REQUEST_FAILED, extra={'reason': response.content.decode('utf-8'),
                                                    'status_code': response.status_code,
                                                    'url': response.url})
    raise exc


def process_unhandled_request(external_logger, exc):
    exc_str = str(exc)
    external_logger.error(traceback.format_exc(), exc_info=True, extra={'exc_str': exc_str})
    raise RuntimeError(REQUEST_FAILED + 'Last failure: {!r}'.format(exc_str))
