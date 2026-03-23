"""
ORM 数据模型 - 模型工厂任务、模型、测试集、评估结果与标签。

设计原则：
- Task.id 与 merges/<task_id> 目录一致，使用 8 位字符串（由调用方传入）。
- 所有时间字段使用 UTC，便于后续分布式与 PostgreSQL 迁移。
"""
from datetime import datetime
import uuid

from app.extensions import db


# -----------------------------------------------------------------------------
# 关联表（多对多）
# -----------------------------------------------------------------------------

model_tags = db.Table(
    "model_tags",
    db.Column("model_id", db.String(36), db.ForeignKey("models.id", ondelete="CASCADE"), primary_key=True),
    db.Column("tag_id", db.String(36), db.ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
    extend_existing=True,
)


# -----------------------------------------------------------------------------
# Task：融合/评估/配方任务，与 merges/<task_id> 一一对应
# -----------------------------------------------------------------------------

class Task(db.Model):
    """
    任务记录，与 merges/<task_id>/metadata.json 双写，后续以 DB 为主。
    id 与目录名一致，为 8 位字符串（由 API 生成）。
    """
    __tablename__ = "tasks"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.String(36), primary_key=True)  # 8 位 task_id，由调用方传入
    status = db.Column(db.String(20), default="pending", nullable=False)  # pending, running, completed, failed, queued
    task_type = db.Column(db.String(50), nullable=False)  # merge, merge_evolutionary, eval_only, recipe_apply
    config = db.Column(db.JSON, nullable=True)  # 任务参数快照

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    started_at = db.Column(db.DateTime, nullable=True)
    finished_at = db.Column(db.DateTime, nullable=True)
    log_path = db.Column(db.String(512), nullable=True)  # 如 merges/<task_id>/ 或 bridge.log 路径

    custom_name = db.Column(db.String(256), nullable=True)
    error = db.Column(db.Text, nullable=True)
    duration_seconds = db.Column(db.Float, nullable=True)
    model_path = db.Column(db.String(1024), nullable=True)

    # 详细时序（供前端展示）
    gen_1_duration = db.Column(db.Float, nullable=True)       # 首代进化耗时（秒）
    avg_merge_time = db.Column(db.Float, nullable=True)       # 单次融合平均耗时（秒）
    final_eval_duration = db.Column(db.Float, nullable=True)   # 最终评估耗时（秒）

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "task_type": self.task_type,
            "config": self.config,
            "custom_name": self.custom_name,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "model_path": self.model_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "log_path": self.log_path,
            "gen_1_duration": self.gen_1_duration,
            "avg_merge_time": self.avg_merge_time,
            "final_eval_duration": self.final_eval_duration,
        }

    def __repr__(self):
        return f"<Task {self.id} {self.status}>"


# -----------------------------------------------------------------------------
# Model：模型注册表（路径、架构、来源、父模型、参数量）
# -----------------------------------------------------------------------------

class Model(db.Model):
    """
    模型基础信息。融合产物在任务成功时写入；基座可后续批量注册或保持仅扫描。
    """
    __tablename__ = "models"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    path = db.Column(db.String(1024), unique=True, nullable=False, index=True)  # 绝对路径
    name = db.Column(db.String(256), nullable=False)
    source = db.Column(db.String(32), nullable=False)  # base | merged | fine_tuned
    parent_model_ids = db.Column(db.JSON, nullable=True)  # 父模型 id 列表（融合时）
    task_id = db.Column(db.String(36), nullable=True, index=True)  # 产出该模型的任务 id

    param_count = db.Column(db.BigInteger, nullable=True)
    architecture = db.Column(db.String(128), nullable=True)  # 如 model_type 或 hidden_size/num_hidden_layers 摘要
    is_vlm = db.Column(db.Boolean, default=False, nullable=True)
    size_bytes = db.Column(db.BigInteger, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    tags = db.relationship("Tag", secondary=model_tags, backref=db.backref("models", lazy="dynamic"))

    def to_dict(self):
        return {
            "id": self.id,
            "path": self.path,
            "name": self.name,
            "source": self.source,
            "parent_model_ids": self.parent_model_ids,
            "task_id": self.task_id,
            "param_count": self.param_count,
            "architecture": self.architecture,
            "is_vlm": self.is_vlm,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "tag_names": [t.name for t in self.tags],
        }

    def __repr__(self):
        return f"<Model {self.name} {self.path}>"


# -----------------------------------------------------------------------------
# TestSet：测试集定义，以 DB 为主，双写 testsets.json
# -----------------------------------------------------------------------------

class TestSet(db.Model):
    """
    测试集元数据。读以 DB 为主；写时双写 DB + testset_repo/data/testsets.json。
    """
    __tablename__ = "testsets"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.String(128), primary_key=True)  # 与原有 testset_id 一致（UUID 或 mmlu-stem 等）
    name = db.Column(db.String(256), nullable=False)
    type = db.Column(db.String(32), nullable=True)   # benchmark | custom
    hf_dataset = db.Column(db.String(256), nullable=True)
    hf_subset = db.Column(db.String(128), nullable=True)
    hf_split = db.Column(db.String(64), nullable=True)
    lm_eval_task = db.Column(db.String(256), nullable=True)
    benchmark_config = db.Column(db.JSON, nullable=True)  # 版本、yaml 路径等
    version = db.Column(db.String(64), nullable=True)     # Benchmark 版本，便于分数可比
    sample_count = db.Column(db.Integer, default=0, nullable=False)
    is_local = db.Column(db.Boolean, default=False, nullable=False)
    local_path = db.Column(db.String(512), nullable=True)
    yaml_template_path = db.Column(db.String(1024), nullable=True)
    created_by = db.Column(db.String(128), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    question_type = db.Column(db.String(64), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def to_dict(self):
        d = {
            "testset_id": self.id,
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "hf_dataset": self.hf_dataset,
            "hf_subset": self.hf_subset,
            "hf_split": self.hf_split,
            "lm_eval_task": self.lm_eval_task,
            "benchmark_config": self.benchmark_config,
            "version": self.version,
            "sample_count": self.sample_count,
            "is_local": self.is_local,
            "local_path": self.local_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if self.yaml_template_path:
            d["yaml_template_path"] = self.yaml_template_path
        if self.created_by:
            d["created_by"] = self.created_by
        if self.notes:
            d["notes"] = self.notes
        if self.question_type:
            d["question_type"] = self.question_type
        return d

    def __repr__(self):
        return f"<TestSet {self.id} {self.name}>"


# -----------------------------------------------------------------------------
# EvaluationResult：单次评估结果，关联 Model 与 TestSet
# -----------------------------------------------------------------------------

class EvaluationResult(db.Model):
    """
    单次评估的指标记录。评估成功后与 _update_leaderboard() 双写（DB + leaderboard.json）。
    """
    __tablename__ = "evaluation_results"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = db.Column(db.String(36), db.ForeignKey("models.id", ondelete="SET NULL"), nullable=True, index=True)
    testset_id = db.Column(db.String(128), db.ForeignKey("testsets.id", ondelete="SET NULL"), nullable=True, index=True)
    task_id = db.Column(db.String(36), nullable=False, index=True)  # 评估任务 id，与 merges/ 一致

    accuracy = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    test_cases = db.Column(db.Integer, nullable=True)
    context = db.Column(db.Integer, nullable=True)
    throughput = db.Column(db.Float, nullable=True)
    time = db.Column(db.Float, nullable=True)  # 评估耗时（秒）
    metrics = db.Column(db.JSON, nullable=True)  # 完整 metrics 快照
    hf_dataset = db.Column(db.String(256), nullable=True)
    hf_subset = db.Column(db.String(128), nullable=True)
    efficiency_score = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    model = db.relationship("Model", backref=db.backref("evaluation_results", lazy="dynamic"))
    testset = db.relationship("TestSet", backref=db.backref("evaluation_results", lazy="dynamic"))

    def to_dict(self):
        return {
            "id": self.id,
            "model_id": self.model_id,
            "testset_id": self.testset_id,
            "task_id": self.task_id,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "test_cases": self.test_cases,
            "context": self.context,
            "throughput": self.throughput,
            "time": self.time,
            "metrics": self.metrics,
            "hf_dataset": self.hf_dataset,
            "hf_subset": self.hf_subset,
            "efficiency_score": self.efficiency_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<EvaluationResult {self.task_id} acc={self.accuracy}>"


# -----------------------------------------------------------------------------
# EvolutionStep：进化搜索每步数据，供 3D 图表与历史查询
# -----------------------------------------------------------------------------

class EvolutionStep(db.Model):
    __tablename__ = "evolution_steps"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    task_id = db.Column(db.String(36), db.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    eval_index = db.Column(db.Integer, nullable=False)
    generation = db.Column(db.Integer, nullable=True)
    accuracy = db.Column(db.Float, nullable=True)
    best_accuracy = db.Column(db.Float, nullable=True)
    genotype = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    task = db.relationship("Task", backref=db.backref("evolution_steps", lazy="dynamic", cascade="all, delete-orphan"))

    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "eval_index": self.eval_index,
            "generation": self.generation,
            "accuracy": self.accuracy,
            "best_accuracy": self.best_accuracy,
            "genotype": self.genotype,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<EvolutionStep task={self.task_id} idx={self.eval_index} acc={self.accuracy}>"


# -----------------------------------------------------------------------------
# Tag：模型属性标签（数学特化、基座、融合产物、待删除、保留等）
# -----------------------------------------------------------------------------

class Tag(db.Model):
    """
    标签表，与 Model 多对多。用于特化分类与优胜劣汰（Keep/Discard）。
    """
    __tablename__ = "tags"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(64), unique=True, nullable=False, index=True)
    category = db.Column(db.String(32), nullable=True)  # specialization | lifecycle | custom
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<Tag {self.name}>"
