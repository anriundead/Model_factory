import logging
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
        from .model_gateway import models as model_gateway_models  # noqa: F401  # 确保 gateway ORM 已加载
        gateway_database_url = app.config.get("MERGEKIT_MODEL_GATEWAY_DATABASE_URL")
        if gateway_database_url:
            from .model_gateway.migration import upgrade_gateway_schema

            upgrade_gateway_schema(root, gateway_database_url)
        else:
            # Local SQLite development keeps the legacy zero-configuration path.
            db.create_all()
        # SQLite 增量列：TestSet.cached_configs / cached_splits（无 Alembic 迁移时）
        try:
            from sqlalchemy import inspect, text

            uri = str(app.config.get("SQLALCHEMY_DATABASE_URI") or "")
            if "sqlite" in uri.lower():
                insp = inspect(db.engine)
                cols = {c["name"] for c in insp.get_columns("testsets")}
                with db.engine.begin() as conn:
                    if "cached_configs" not in cols:
                        conn.execute(text("ALTER TABLE testsets ADD COLUMN cached_configs TEXT"))
                    if "cached_splits" not in cols:
                        conn.execute(text("ALTER TABLE testsets ADD COLUMN cached_splits TEXT"))
        except Exception as ex:
            logging.getLogger("mergeKit_beta").warning("testsets 表增量列检查跳过: %s", ex)
        if not getattr(_admin, "_views_registered", False):
            from .admin import register_admin_views
            register_admin_views(_admin)
            _admin._views_registered = True
        try:
            from .model_gateway.runtime import (
                mark_services_stopped_after_restart,
                recover_inflight_requests_after_restart,
                should_recover_services_on_start,
            )
            if should_recover_services_on_start():
                recovered = mark_services_stopped_after_restart(db.session)
                if recovered:
                    logging.getLogger("mergeKit_beta").info("model gateway 重启恢复为手动管理: %d 个", recovered)
                recovered_requests = recover_inflight_requests_after_restart(db.session)
                if recovered_requests:
                    logging.getLogger("mergeKit_beta").info(
                        "model gateway 在途请求已终结且未重放: %d 个", recovered_requests
                    )
        except Exception as ex:
            logging.getLogger("mergeKit_beta").warning("model gateway 状态恢复跳过: %s", ex)

    state = AppState()
    setup_logging(app, state.config)
    state.logger = app.logger
    services = Services(state)
    services.app = app  # 供 Worker 内 DB 写入使用 app_context
    dataset_service = DatasetInfoService(state.config)
    register_routes(app, state, services, dataset_service)
    if getattr(Config, "MERGEKIT_MODEL_GATEWAY_ENABLED", True):
        from .model_gateway import register_model_gateway
        register_model_gateway(app)
        from .model_gateway.research_job_worker import start_research_worker
        start_research_worker(app)
    services.start_task_worker()
    # 启动时全量同步 models 表：删除磁盘不存在的记录，写入新扫描到的基座/融合模型
    with app.app_context():
        try:
            stats = services.sync_models_db_from_disk()
            import logging
            logging.getLogger("mergeKit_beta").info(
                "启动时 models 表已同步: removed=%s base=%s merged=%s",
                stats.get("removed", 0),
                stats.get("upserted_base", 0),
                stats.get("upserted_merged", 0),
            )
        except Exception as e:
            import logging
            logging.getLogger("mergeKit_beta").warning("启动时 models 表同步跳过: %s", e)
        try:
            from .model_publication import reconcile_publications
            from .repositories import model_register_published, publication_task_is_active

            recovery = reconcile_publications(
                Config.PUBLISHED_MODELS_PATH,
                model_register_published,
                active_check_fn=publication_task_is_active,
            )
            logging.getLogger("mergeKit_beta").info(
                "启动时正式模型已恢复: registered=%s quarantined=%s staging_cleaned=%s",
                recovery.get("registered", 0),
                recovery.get("quarantined", 0),
                recovery.get("staging_cleaned", 0),
            )
        except Exception as e:
            import logging
            logging.getLogger("mergeKit_beta").warning("启动时正式模型恢复跳过: %s", e)
    return app


# CLI 脚本（如 scripts/report_local_model_pairing.py）在 import 子模块前设置 MERGEKIT_CLI_SCRIPT=1，避免此处启动 Flask 与 Worker。
if os.environ.get("MERGEKIT_CLI_SCRIPT") == "1":
    app = None
else:
    app = create_app()
