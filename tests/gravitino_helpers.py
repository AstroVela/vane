# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Process-isolated Gravitino/S3 HTTP fixture for native bind-time I/O."""

import copy
import gzip
import hashlib
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlsplit
from xml.etree.ElementTree import Element, SubElement, tostring


def serve_gravitino(state, ready):
    uploads = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def handle_request(self):
            if urlsplit(self.path).path.startswith("/bucket/") or urlsplit(self.path).path == "/bucket":
                return self.object_request()
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length)) if length else None
            state["requests"].append((self.command, self.path, payload))
            if self.headers.get("Authorization") != "Bearer fixture-token":
                return self.respond(401, {"code": 1001})
            if state["fault"]:
                fault = state["fault"]
                if fault == "oversized":
                    return self.respond(200, {"code": 0, "padding": "x" * 16384})
                if fault == "compressed":
                    return self.respond(200, {"code": 0, "padding": "x" * 16384}, compressed=True)
                if fault == "redirect":
                    self.send_response(302)
                    self.send_header("Location", "/should-not-follow")
                    self.end_headers()
                    return
                if fault == "slow":
                    time.sleep(0.25)
                if fault == "failed-write" and self.command != "GET":
                    return self.respond(503, {"code": 1001})
                if fault == "invalid-json":
                    return self.respond(200, [])
                if fault == "missing-outcome":
                    return self.respond(200, {"code": 0})
                if fault == "not-dropped":
                    return self.respond(200, {"code": 0, "dropped": False})
            path = [unquote(part) for part in urlsplit(self.path).path.split("/") if part]
            if discovery := state.get("discovery"):
                return self.discovery_request(path, discovery)
            if path[:6] != ["prefix", "api", "metalakes", "lake", "catalogs", "media"]:
                return self.respond(404, {"code": 1001})
            path = path[6:]
            if not path:
                if self.command == "PUT" and not self.alter(state["catalog"], payload, "catalog"):
                    return
                return self.respond(200, {"code": 0, "catalog": state["catalog"]})
            if path[0] != "schemas":
                return self.respond(404, {"code": 1001})
            if len(path) == 1:
                if self.command == "POST":
                    if payload["name"] in state["schemas"]:
                        return self.respond(409, {"code": 1001})
                    state["schemas"][payload["name"]] = payload
                    return self.respond(200, {"code": 0, "schema": payload})
                return self.respond(200, {"code": 0, "identifiers": [{"name": n} for n in state["schemas"].keys()]})
            schema = path[1]
            if schema not in state["schemas"]:
                return self.respond(404, {"code": 1001})
            if len(path) == 2:
                if self.command == "DELETE":
                    children = [key for key in state["filesets"].keys() if key[0] == schema]
                    if children and "cascade=true" not in self.path:
                        return self.respond(409, {"code": 1001})
                    for key in children:
                        del state["filesets"][key]
                    del state["schemas"][schema]
                    return self.respond(200, {"code": 0, "dropped": True})
                if self.command == "PUT":
                    data = state["schemas"][schema]
                    if not self.alter(data, payload, "schema"):
                        return
                    state["schemas"][schema] = data
                return self.respond(200, {"code": 0, "schema": state["schemas"][schema]})
            if path[2] != "filesets":
                return self.respond(404, {"code": 1001})
            if len(path) == 3:
                if self.command == "POST":
                    key = (schema, payload["name"])
                    if key in state["filesets"]:
                        return self.respond(409, {"code": 1001})
                    data = copy.deepcopy(payload)
                    if "storageLocation" in data:
                        data["storageLocations"] = {"unknown": data["storageLocation"]}
                    state["filesets"][key] = data
                    return self.respond(200, {"code": 0, "fileset": data})
                names = [{"name": n} for s, n in state["filesets"].keys() if s == schema]
                return self.respond(200, {"code": 0, "identifiers": names})
            key = (schema, path[3])
            if key not in state["filesets"]:
                return self.respond(404, {"code": 1001})
            data = state["filesets"][key]
            if self.command == "DELETE":
                del state["filesets"][key]
                return self.respond(200, {"code": 0, "dropped": True})
            if self.command == "PUT":
                if not self.alter(data, payload, "fileset"):
                    return
                state["filesets"][key] = data
                if key[1] != data["name"]:
                    state["filesets"][(schema, data["name"])] = state["filesets"].pop(key)
            self.respond(200, {"code": 0, "fileset": data})

        def discovery_request(self, path, config):
            if self.command != "GET":
                return self.respond(405, {"code": 1001})
            if path[:4] != ["prefix", "api", "metalakes", config["metalake"]]:
                return self.respond(404, {"code": 1001})
            if len(path) == 4:
                return self.discovery_response({"code": 0, "metalake": {"name": config["metalake"]}})
            if path[4] != "catalogs":
                return self.respond(404, {"code": 1001})
            if len(path) == 5:
                return self.discovery_response(
                    {
                        "code": 0,
                        "identifiers": [
                            {"namespace": [config["metalake"]], "name": name} for name in config["catalogs"]
                        ],
                    }
                )
            catalog = path[5]
            if catalog not in config["catalogs"]:
                return self.respond(404, {"code": 1001})
            if len(path) == 6:
                return self.discovery_response({"code": 0, "catalog": config["catalogs"][catalog]})
            if path[6] != "schemas":
                return self.respond(404, {"code": 1001})
            schemas = config["schemas"].get(catalog, {})
            parent = [config["metalake"], catalog]
            if len(path) == 7:
                return self.discovery_response(
                    {"code": 0, "identifiers": [{"namespace": parent, "name": name} for name in schemas]}
                )
            schema = path[7]
            if schema not in schemas:
                return self.respond(404, {"code": 1001})
            if len(path) == 8:
                return self.discovery_response({"code": 0, "schema": schemas[schema]})
            if path[8] != "tables":
                return self.respond(404, {"code": 1001})
            tables = config["tables"].get((catalog, schema), {})
            if len(path) == 9:
                return self.discovery_response(
                    {"code": 0, "identifiers": [{"namespace": parent + [schema], "name": name} for name in tables]}
                )
            if len(path) == 10 and path[9] in tables:
                return self.discovery_response({"code": 0, "table": tables[path[9]]})
            return self.respond(404, {"code": 1001})

        def discovery_response(self, body):
            fault = state.get("discovery_fault")
            body = copy.deepcopy(body)
            if fault == "not-found":
                return self.respond(404, {"code": 1001})
            if fault == "forbidden":
                return self.respond(403, {"code": 1001})
            key = "identifiers" if "identifiers" in body else None
            if key:
                namespace = body[key][0]["namespace"] if body[key] else []
                if fault == "missing-list":
                    del body[key]
                elif fault == "bad-list":
                    body[key] = {}
                elif fault == "too-many":
                    body[key] = [{"namespace": namespace, "name": f"name{i}"} for i in range(4097)]
                elif fault == "duplicate" and body[key]:
                    body[key].append(body[key][0])
                elif fault == "bad-entry":
                    body[key] = [123]
                elif fault == "bad-name":
                    body[key] = [{"namespace": namespace, "name": "../wrong"}]
                elif fault == "bad-namespace":
                    body[key][0]["namespace"] = ["different", "catalog"]
                elif fault == "pagination":
                    body["nextPageToken"] = "another-page"
                elif fault == "has-more":
                    body["hasMore"] = True
            for kind in ("metalake", "catalog", "schema", "table"):
                if kind in body:
                    if fault == "wrong-resource":
                        body[kind]["name"] = "not-the-requested-resource"
                    elif fault == "missing-resource-name":
                        del body[kind]["name"]
                    elif fault == "missing-provider" and kind == "catalog":
                        del body[kind]["provider"]
                    elif fault == "bad-properties":
                        body[kind]["properties"] = ["not-an-object"]
            return self.respond(200, body)

        def object_request(self):
            state["object_requests"].append((self.command, self.headers.get("Authorization")))
            request = urlsplit(self.path)
            objects = {"object.txt": state["object"], **dict(state["objects"])}
            query = parse_qs(request.query, keep_blank_values=True)
            key = unquote(request.path.removeprefix("/bucket/"))
            state["object_paths"].append((self.command, key))
            fault = state["object_fault"]
            if fault == "marker-unknown" and key.endswith("/committed"):
                if self.command == "PUT":
                    state["objects"][key] = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if key in state["objects"]:
                    return self.respond(503, {"error": "injected marker response/readback failure"})
            if fault == "manifest-write" and key.endswith("/manifest.txt") and self.command in ("PUT", "POST"):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                return self.respond(503, {"error": "injected manifest publication failure"})
            if (
                fault == "result-read"
                and key.endswith("/manifest.txt")
                and self.command == "GET"
                and key.removesuffix("manifest.txt") + "committed" in state["objects"]
            ):
                return self.respond(503, {"error": "injected post-commit manifest read failure"})
            if self.command in ("PUT", "POST", "DELETE"):
                length = int(self.headers.get("Content-Length", "0"))
                payload = self.rfile.read(length)
                upload_id = query.get("uploadId", [""])[0]
                if self.command == "POST" and "uploads" in query:
                    upload_id = uuid.uuid4().hex
                    uploads[upload_id] = {}
                    root = Element("InitiateMultipartUploadResult")
                    SubElement(root, "Bucket").text = "bucket"
                    SubElement(root, "Key").text = key
                    SubElement(root, "UploadId").text = upload_id
                elif self.command == "PUT" and upload_id:
                    uploads[upload_id][int(query["partNumber"][0])] = payload
                    root = None
                elif self.command == "POST" and upload_id:
                    parts = uploads.pop(upload_id)
                    payload = b"".join(parts[number] for number in sorted(parts))
                    state["objects"][key] = payload
                    root = Element("CompleteMultipartUploadResult")
                    SubElement(root, "Bucket").text = "bucket"
                    SubElement(root, "Key").text = key
                    SubElement(root, "ETag").text = '"fixture-etag"'
                elif self.command == "PUT":
                    state["objects"][key] = payload
                    root = None
                else:
                    if upload_id:
                        uploads.pop(upload_id, None)
                    else:
                        state["objects"].pop(key, None)
                    root = None
                if fault == "pause-data" and key.endswith((".parquet", ".csv")) and key in state["objects"]:
                    state["object_blocked"] = key
                    deadline = time.monotonic() + 30
                    while state["object_fault"] == "pause-data" and time.monotonic() < deadline:
                        time.sleep(0.02)
                body = tostring(root) if root is not None else b""
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("ETag", '"' + hashlib.md5(payload, usedforsecurity=False).hexdigest() + '"')
                self.end_headers()
                self.wfile.write(body)
                return
            if "list-type" in query:
                root = Element("ListBucketResult")
                SubElement(root, "IsTruncated").text = "false"
                SubElement(root, "EncodingType").text = "url"
                prefix = query.get("prefix", [""])[0]
                for key, value in sorted(objects.items()):
                    if key.startswith(prefix):
                        item = SubElement(root, "Contents")
                        SubElement(item, "Key").text = quote(key, safe="/")
                        SubElement(item, "Size").text = str(len(value))
                        SubElement(item, "LastModified").text = "2026-09-29T00:00:00.000Z"
                body = tostring(root)
            else:
                if key not in objects:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = objects[key]
            bounds = self.headers.get("Range")
            self.send_response(206 if bounds else 200)
            if bounds:
                start, end = map(int, bounds.removeprefix("bytes=").split("-"))
                end = min(end, len(body) - 1)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
                body = body[start : end + 1]
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def alter(self, data, payload, resource):
            # Gravitino 1.3's Schema/Catalog/FilesetUpdateRequest subtypes differ.
            allowed = {
                "schema": {"setProperty", "removeProperty"},
                "catalog": {"rename", "updateComment", "setProperty", "removeProperty"},
                "fileset": {"rename", "updateComment", "removeComment", "setProperty", "removeProperty"},
            }[resource]
            if any(change["@type"] not in allowed for change in payload["updates"]):
                self.respond(400, {"code": 1001})
                return False
            for change in payload["updates"]:
                if change["@type"] == "rename":
                    data["name"] = change["newName"]
                elif change["@type"] == "updateComment":
                    data["comment"] = change["newComment"]
                elif change["@type"] == "removeComment":
                    data["comment"] = None
                elif change["@type"] == "setProperty":
                    properties = dict(data["properties"])
                    properties[change["property"]] = change["value"]
                    data["properties"] = properties
                elif change["@type"] == "removeProperty":
                    properties = dict(data["properties"])
                    properties.pop(change["property"], None)
                    data["properties"] = properties
            return True

        def respond(self, status, value, *, compressed=False):
            data = json.dumps(value, ensure_ascii=False, default=lambda proxy: proxy._getvalue()).encode()
            if compressed:
                data = gzip.compress(data)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = handle_request

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    ready.send(server.server_port)
    ready.close()
    try:
        server.serve_forever()
    finally:
        server.server_close()
