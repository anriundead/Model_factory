from flask_admin.contrib.sqla import ModelView
from app.extensions import db, admin
from app.models import Task

class TaskView(ModelView):
    column_list = ('id', 'status', 'task_type', 'created_at', 'updated_at')
    column_searchable_list = ('id', 'status', 'task_type')
    column_filters = ('status', 'task_type', 'created_at')
    can_create = True
    can_edit = True
    can_delete = True
    page_size = 50

def register_admin_views():
    admin.add_view(TaskView(Task, db.session))
