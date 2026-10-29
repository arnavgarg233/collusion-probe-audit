#!/usr/bin/env python3
"""Check every number the paper states against the stored records in results/tables/."""

from collusion_probe_audit.cli import verify_main

if __name__ == "__main__":
    raise SystemExit(verify_main())
