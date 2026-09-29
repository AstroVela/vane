# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Process-isolated Gravitino/S3 HTTP fixture for native bind-time I/O."""

import copy
import gzip
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlsplit
from xml.etree.ElementTree import Element, SubElement, tostring


def serve_gravitino(state, ready):
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

        def object_request(self):
            state["object_requests"].append((self.command, self.headers.get("Authorization")))
            request = urlsplit(self.path)
            objects = {"object.txt": state["object"], **dict(state["objects"])}
            query = parse_qs(request.query)
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
                key = unquote(request.path.removeprefix("/bucket/"))
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
