from app.extensions import db
from datetime import datetime
import uuid

class Task(db.Model):
    """
    任务模型，用于替代或补充 metadata.json
    """
    __tablename__ = 'tasks'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status = db.Column(db.String(20), default='pending')  # pending, running, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 存储任务配置，使用 JSON 类型 (SQLite 中会自动转为 Text)
    config = db.Column(db.JSON, nullable=True)
    
    # 任务类型: merge, eval, recipe
    task_type = db.Column(db.String(50), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'task_type': self.task_type,
            'config': self.config
        }

    def __repr__(self):
        return f'<Task {self.id} {self.status}>'
