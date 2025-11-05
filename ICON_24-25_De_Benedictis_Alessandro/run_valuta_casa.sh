#!/bin/bash
cd "$(dirname "$0")"
export SYSTEM_VERSION_COMPAT=0
.venv/bin/python ImmoValuta.py
