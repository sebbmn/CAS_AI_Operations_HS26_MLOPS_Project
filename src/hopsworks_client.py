"""Single place where we log in to Hopsworks."""
import sys

import hopsworks

from config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT


def ensure_credentials():
    """Fail fast (before any work is done) if the API key is missing."""
    if not HOPSWORKS_API_KEY or HOPSWORKS_API_KEY.startswith("REPLACE_WITH"):
        sys.exit(
            "ERROR: HOPSWORKS_API_KEY is not set.\n"
            "Copy .env.example to .env and insert your Hopsworks API key "
            "(hopsworks.ai -> Account Settings -> API keys)."
        )


def login():
    ensure_credentials()
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT or None,
    )
    print(f"Connected to Hopsworks project '{project.name}'")
    return project
