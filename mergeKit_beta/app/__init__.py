import os

from flask import Flask

from config import Config
from .state import AppState
from .logging_config import setup_logging
from .services import Services
from .dataset_info import DatasetInfoService
from .routes import register_routes
from .extensions import db, migrate, admin

def create_app():
    Config.setup_environment()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app = Flask(__name__, static_folder=os.path.join(root, "static"), template_folder=os.path.join(root, "templates"))
    
    # 加载配置
    app.config.from_object(Config)
    
    # 初始化数据库
    db.init_app(app)
    migrate.init_app(app, db)
    admin.init_app(app)
    
    # 注册模型和后台视图
    with app.app_context():
        from . import models
        db.create_all()
        from .admin import register_admin_views
        register_admin_views()

    state = AppState()
    setup_logging(app, state.config)
    state.logger = app.logger
    services = Services(state)
    dataset_service = DatasetInfoService(state.config)
    register_routes(app, state, services, dataset_service)
    services.start_task_worker()
    return app


app = create_app()
