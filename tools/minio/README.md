# Local MinIO test service

Set explicit development-only credentials before starting the service:

```bash
export MINIO_ROOT_USER='choose-a-local-access-key'
export MINIO_ROOT_PASSWORD='choose-a-long-local-password'
tools/minio/up.sh
```

Do not reuse production credentials. The scripts bind the API and console to local ports by default; review Docker and firewall settings before exposing them to another host.

To mirror a directory into the local `datasets` bucket:

```bash
tools/minio/import-data.sh /path/to/data
```

Stop the service with `tools/minio/down.sh`.
