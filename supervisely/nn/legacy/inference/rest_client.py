import argparse
import io
import json
import requests

from supervisely.io import fs as sly_fs
from supervisely.nn.inference.rest_constants import GET_OUTPUT_META, IMAGE, INFERENCE, MODEL, SUPPORTED_REQUEST_TYPES

from requests_toolbelt import MultipartEncoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference REST client for standalone Supervisely models.')
    parser.add_argument('--server-url', required=True)
    parser.add_argument('--request-type', required=True, choices=SUPPORTED_REQUEST_TYPES)
    parser.add_argument('--in-image', default='')
    parser.add_argument('--out-json', default='')
    args = parser.parse_args()

    request_url = args.server_url + '/' + MODEL + '/' + args.request_type

    response_json = None
    if args.request_type == GET_OUTPUT_META:
        response = requests.post(request_url)
    elif args.request_type == INFERENCE:
        with open(args.in_image, 'rb') as fin:
            img_bytes = fin.read()
        img_ext = sly_fs.get_file_ext(args.in_image)
        encoder = MultipartEncoder({IMAGE: (args.in_image, io.BytesIO(img_bytes), 'application/octet-stream')})
        response = requests.post(request_url, data=encoder, headers={'Content-Type': encoder.content_type})
    else:
        raise ValueError(
            'Unknown model request type: {!r}. Only the following request types are supported: {!r}.'.format(
                args.request_type, SUPPORTED_REQUEST_TYPES))

    response.raise_for_status()
    response_str = json.dumps(response.json(), indent=4, sort_keys=True)

    if args.out_json:
        with open(args.out_json, 'w') as fout:
            fout.write(response_str)
    else:
        print(response_str)
