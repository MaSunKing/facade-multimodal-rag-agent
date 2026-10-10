"""SQLite journal policy for state shared between Windows and WSL."""
import os
import re


def state_journal_mode(path, *, platform=None):
    value = str(path).replace('\\', '/')
    # WAL shared memory is not portable across Windows/DrvFS clients.
    if (platform or os.name) == 'nt' or re.match(r'^/mnt/[a-z](?:/|$)', value, re.I):
        return 'DELETE'
    return 'WAL'
