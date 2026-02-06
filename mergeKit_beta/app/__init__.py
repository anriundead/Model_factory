import os

from flask import Flask

from config import Config
from .state import AppState
from .logging_config import setup_logging
from .services import Services
from .dataset_info import DatasetInfoService
from .routes import register_routes


def create_app():
    Config.setup_environment()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app = Flask(__name__, static_folder=os.path.join(root, "static"), template_folder=os.path.join(root, "templates"))
    state = AppState()
    setup_logging(app, state.config)
    state.logger = app.logger
    services = Services(state)
    dataset_service = DatasetInfoService(state.config)
    register_routes(app, state, services, dataset_service)
    services.start_task_worker()
    return app


app = create_app()
