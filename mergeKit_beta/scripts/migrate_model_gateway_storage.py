#!/usr/bin/env python3
"""Copy legacy Gateway tables to a separately configured database."""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MERGEKIT_CLI_SCRIPT", "1")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sqlalchemy import create_engine  # noqa: E402

from app.model_gateway import models  # noqa: F401,E402
from app.model_gateway.migration import copy_legacy_gateway_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Legacy SQLAlchemy URL, usually SQLite")
    parser.add_argument("--target", required=True, help="Gateway PostgreSQL SQLAlchemy URL")
    args = parser.parse_args()
    source = create_engine(args.source)
    target = create_engine(args.target)
    from app.extensions import db

    db.metadatas["model_gateway"].create_all(target)
    for table, count in copy_legacy_gateway_rows(source, target).items():
        print(f"{table}: copied={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
