#!/bin/bash

# TODO: check that wireguard-tools is installed

# ./sly-net.sh up XXX https://app.supervise.ly .
# ./sly-net.sh <up|down> <token> <server_address> <config and keys folder>

set -u
set -e
set -o pipefail
FOLDER="${4:-.}"
SLY_NET_SERVER="${3:-https://app.supervise.ly}"
cd "$FOLDER"
cat <<EOT > wg0.conf
[Interface]
PrivateKey = __PRIVATE_KEY__
Address = __IP__/16
[Peer]
PublicKey = __SERVER_PUBLIC_KEY__
AllowedIPs = 10.8.0.0/16
Endpoint = __SERVER_ENDPOINT__
PersistentKeepalive = 25
EOT
if [ ! -f "${FOLDER}/public.key" ]; then
    umask 077
    wg genkey > private.key 
    cat private.key | wg pubkey > public.key
fi

chmod 600 wg0.conf

RESPONSE=$(curl -s --show-error --fail -X POST "${SLY_NET_SERVER}/net/register/${2}/$(cat public.key)")
RESPONSE_ARR=(${RESPONSE//;/ })
echo "s#__IP__#${RESPONSE_ARR[0]}#g"
sed -i -E "s#__IP__#${RESPONSE_ARR[0]}#g" wg0.conf
sed -i -E "s#__SERVER_PUBLIC_KEY__#${RESPONSE_ARR[1]}#g" wg0.conf
sed -i -E "s#__SERVER_ENDPOINT__#${RESPONSE_ARR[2]}#g" wg0.conf
sed -i -E "s#__PRIVATE_KEY__#$(cat private.key)#g" wg0.conf
wg-quick "$1" "${FOLDER}/wg0.conf"
