from flask_admin.contrib.sqla import ModelView
from app.extensions import db, admin
from app.models import Task, Model, TestSet, EvaluationResult, Tag

# 防止 start_app.sh 下 app 包被加载两次时对同一 Flask app 重复注册 Admin 视图
_registered_admin_app_ids = set()


class TaskView(ModelView):
    column_list = ("id", "status", "task_type", "created_at", "updated_at", "started_at", "finished_at")
    column_searchable_list = ("id", "status", "task_type")
    column_filters = ("status", "task_type", "created_at")
    can_create = True
    can_edit = True
    can_delete = True
    page_size = 50


class ModelView_(ModelView):
    column_list = ("id", "name", "path", "source", "task_id", "created_at")
    column_searchable_list = ("name", "path")
    column_filters = ("source",)
    can_create = True
    can_edit = True
    can_delete = True
    page_size = 50


class TestSetView(ModelView):
    column_list = ("id", "name", "hf_dataset", "hf_subset", "sample_count", "created_at")
    column_searchable_list = ("id", "name", "hf_dataset")
    column_filters = ("type",)
    can_create = True
    can_edit = True
    can_delete = True
    page_size = 50


class EvaluationResultView(ModelView):
    column_list = ("id", "task_id", "model_id", "testset_id", "accuracy", "test_cases", "created_at")
    column_searchable_list = ("task_id",)
    column_filters = ("testset_id",)
    can_create = False
    can_edit = True
    can_delete = True
    page_size = 50


class TagView(ModelView):
    column_list = ("id", "name", "category", "created_at")
    column_searchable_list = ("name",)
    column_filters = ("category",)
    can_create = True
    can_edit = True
    can_delete = True
    page_size = 50


def register_admin_views(admin_instance=None):
    a = admin_instance if admin_instance is not None else admin
    app = getattr(a, "_app", None) or getattr(a, "app", None)
    if app is not None and id(app) in _registered_admin_app_ids:
        return
    if app is not None:
        _registered_admin_app_ids.add(id(app))
    a.add_view(TaskView(Task, db.session, name="Task", category="Tasks", endpoint="admin_task"))
    a.add_view(ModelView_(Model, db.session, name="Model", category="Models", endpoint="admin_model"))
    a.add_view(TestSetView(TestSet, db.session, name="TestSet", category="TestSets", endpoint="admin_testset"))
    a.add_view(EvaluationResultView(EvaluationResult, db.session, name="EvaluationResult", category="Eval", endpoint="admin_evalresult"))
    a.add_view(TagView(Tag, db.session, name="Tag", category="Tags", endpoint="admin_tag"))
