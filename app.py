from __future__ import annotations

import gradio_client.utils as gradio_client_utils

from age_gender_predictor.config import APP_HOST, APP_PORT, APP_SHARE
from age_gender_predictor.ui import create_app


def patch_gradio_schema_bug() -> None:
    original = gradio_client_utils._json_schema_to_python_type

    def safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return original(schema, defs)

    gradio_client_utils._json_schema_to_python_type = safe_json_schema_to_python_type


patch_gradio_schema_bug()
app = create_app()


def launch_app() -> None:
    try:
        app.launch(server_name=APP_HOST, server_port=APP_PORT, share=APP_SHARE, show_api=False)
    except OSError as error:
        if "Cannot find empty port" not in str(error):
            raise
        app.launch(server_name=APP_HOST, server_port=None, share=APP_SHARE, show_api=False)


if __name__ == "__main__":
    launch_app()
