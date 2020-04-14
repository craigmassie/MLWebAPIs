#!/bin/bash

if [ -f .env ]
then
  export $(cat /app/.env | sed 's/#.*//g' | xargs)
fi

# Manual IP resolution for Azure Storage endpoint, as recommended here:
# https://github.com/epam/cloud-pipeline/issues/61#issuecomment-480773072
hostname=${AZURE_STORAGE_ACCOUNT}.blob.core.windows.net 
ip=`dig +short $hostname | tail -1`
if [ -n "$ip" ]; then
    echo IP: $ip
    echo "$ip  $hostname" >> /etc/hosts
else
    echo Could not resolve hostname.
fi

set -euo pipefail
set -o errexit
set -o errtrace
IFS=$'\n\t'

# mount our blobstore
test ${AZURE_MOUNT_POINT}
# rm -rf ${AZURE_MOUNT_POINT}
mkdir -p ${AZURE_MOUNT_POINT}

blobfuse ${AZURE_MOUNT_POINT} --use-https=true --tmp-path=/tmp/blobfuse/${AZURE_STORAGE_ACCOUNT} --container-name=${AZURE_STORAGE_ACCOUNT_CONTAINER} -o allow_other

# run the command passed to us
exec "$@"