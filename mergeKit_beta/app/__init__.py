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
    from .extensions import admin as _admin
    _admin.init_app(app)

    # 注册模型和后台视图（仅注册一次，避免 CLI 与模块加载重复调用 create_app 时重复注册）
    with app.app_context():
        from . import models  # noqa: F401  # 确保 ORM 已加载
        # 方案 B：先执行迁移，由迁移统一建表（不再依赖 db.create_all）
        try:
            from flask_migrate import upgrade
            upgrade()
        except Exception as e:
            import logging
            logging.getLogger("mergeKit_beta").warning("启动时自动迁移跳过: %s", e)
        if not getattr(_admin, "_views_registered", False):
            from .admin import register_admin_views
            register_admin_views(_admin)
            _admin._views_registered = True

    state = AppState()
    setup_logging(app, state.config)
    state.logger = app.logger
    services = Services(state)
    services.app = app  # 供 Worker 内 DB 写入使用 app_context
    dataset_service = DatasetInfoService(state.config)
    register_routes(app, state, services, dataset_service)
    services.start_task_worker()
    # 启动时同步「文件/merges → models 表」+ 扫描基座模型入 DB
    with app.app_context():
        try:
            services.model_repo_list()
        except Exception as e:
            import logging
            logging.getLogger("mergeKit_beta").warning("启动时 models 同步跳过: %s", e)
        try:
            n = services.scan_base_models_to_db()
            if n:
                import logging
                logging.getLogger("mergeKit_beta").info("启动时扫描基座模型入 DB: %d 条", n)
        except Exception as e:
            import logging
            logging.getLogger("mergeKit_beta").warning("启动时基座模型扫描跳过: %s", e)
    return app


app = create_app()
