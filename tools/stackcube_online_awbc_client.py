#!/usr/bin/env python3
"""Send one JSON command to a resident StackCube online AWBC worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--args-json", default="{}")
    args = parser.parse_args()
    request = {"command": args.command, "args": json.loads(args.args_json)}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(args.socket))
        client.sendall((json.dumps(request) + "\n").encode())
        response = client.makefile("r", encoding="utf-8").readline()
    if not response:
        raise RuntimeError("worker closed the socket without a response")
    payload = json.loads(response)
    print(json.dumps(payload, indent=2))
    if not payload.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
