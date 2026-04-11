"""
数据库写入层：任务、模型、测试集、评估结果的创建与更新。

所有函数均需在 Flask 应用上下文中调用（如 request 或 worker 线程内已 push 的 app context）。
"""
from datetime import datetime

from app.extensions import db
from app.models import Task, Model, TestSet, EvaluationResult, Tag, EvolutionStep


# -----------------------------------------------------------------------------
# Task
# -----------------------------------------------------------------------------

def task_upsert(task_id: str, task_type: str, config: dict) -> Task:
    """创建或更新任务记录（提交时调用）。id 为 8 位 task_id。"""
    task = db.session.get(Task, task_id)
    if task is None:
        task = Task(id=task_id, status="queued", task_type=task_type, config=config)
        db.session.add(task)
    else:
        task.status = task.status or "queued"
        task.config = config
        task.updated_at = datetime.utcnow()
    db.session.commit()
    return task


def task_mark_running(task_id: str, log_path: str = None):
    """标记任务为运行中。"""
    task = db.session.get(Task, task_id)
    if task:
        task.status = "running"
        task.started_at = datetime.utcnow()
        if log_path:
            task.log_path = log_path
        task.updated_at = datetime.utcnow()
        db.session.commit()


def task_update_after_completion(
    task_id: str,
    status: str,
    gen_1_duration: float = None,
    avg_merge_time: float = None,
    final_eval_duration: float = None,
):
    """任务结束后更新状态与时序字段。status: completed | failed."""
    task = db.session.get(Task, task_id)
    if task:
        task.status = status
        task.finished_at = datetime.utcnow()
        task.updated_at = datetime.utcnow()
        if gen_1_duration is not None:
            task.gen_1_duration = gen_1_duration
        if avg_merge_time is not None:
            task.avg_merge_time = avg_merge_time
        if final_eval_duration is not None:
            task.final_eval_duration = final_eval_duration
        db.session.commit()


def task_backfill_from_metadata(task_id: str, meta: dict) -> Task:
    """从 metadata.json 内容回填/更新任务记录（迁移脚本用）。存在则更新，不存在则创建。"""
    status_map = {"success": "completed", "error": "failed"}
    raw_status = meta.get("status") or "pending"
    status = status_map.get(raw_status, raw_status)
    task_type = meta.get("type") or "merge"
    config = {k: v for k, v in meta.items() if k not in (
        "id", "status", "type", "custom_name", "error", "duration_seconds",
        "output_path", "model_path", "created_at", "updated_at", "started_at", "finished_at",
    )}
    if meta.get("output_path"):
        config["output_path"] = meta["output_path"]
    if meta.get("created_at"):
        config["created_at"] = meta["created_at"]

    task = db.session.get(Task, task_id)
    if task is None:
        task = Task(
            id=task_id,
            status=status,
            task_type=task_type,
            config=config,
            custom_name=meta.get("custom_name"),
            error=meta.get("error"),
            duration_seconds=meta.get("duration_seconds"),
            model_path=meta.get("model_path") or meta.get("output_path"),
        )
        db.session.add(task)
    else:
        task.status = status
        task.task_type = task_type
        task.config = config
        task.custom_name = meta.get("custom_name") or task.custom_name
        task.error = meta.get("error")
        task.duration_seconds = meta.get("duration_seconds")
        task.model_path = meta.get("model_path") or meta.get("output_path") or task.model_path
        task.updated_at = datetime.utcnow()
        if status in ("completed", "failed") and not task.finished_at:
            task.finished_at = datetime.utcnow()
    db.session.commit()
    return task


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

def model_register(
    path: str,
    name: str,
    source: str = "merged",
    task_id: str = None,
    parent_model_ids: list = None,
    param_count: int = None,
    architecture: str = None,
    is_vlm: bool = None,
    size_bytes: int = None,
) -> Model:
    """注册或更新模型。path 唯一，存在则更新 name 等。"""
    path = path.rstrip("/")
    model = db.session.query(Model).filter_by(path=path).first()
    if model is None:
        model = Model(
            path=path,
            name=name,
            source=source,
            task_id=task_id,
            parent_model_ids=parent_model_ids or [],
            param_count=param_count,
            architecture=architecture,
            is_vlm=is_vlm,
            size_bytes=size_bytes,
        )
        db.session.add(model)
    else:
        model.name = name
        model.source = source
        # 列表同步等场景可能解析不到 metadata，勿用 None 覆盖 worker 已写入的 task_id
        if task_id is not None:
            model.task_id = task_id
        if parent_model_ids is not None:
            model.parent_model_ids = parent_model_ids
        if param_count is not None:
            model.param_count = param_count
        if architecture is not None:
            model.architecture = architecture
        if is_vlm is not None:
            model.is_vlm = is_vlm
        if size_bytes is not None:
            model.size_bytes = size_bytes
    db.session.commit()
    if source == "merged":
        model_add_tag(model, "融合产物", category="lifecycle")
    return model


def model_get_by_path(path: str) -> Model | None:
    """按路径查询模型。"""
    path = path.rstrip("/")
    return db.session.query(Model).filter_by(path=path).first()


def model_get_by_id(model_id: str) -> Model | None:
    """按 id 查询模型。"""
    if not (model_id or "").strip():
        return None
    return db.session.get(Model, model_id.strip())


def model_delete_by_path(path: str) -> bool:
    """按路径删除模型记录。EvaluationResult.model_id 为 SET NULL，可安全删除。"""
    path = (path or "").strip().rstrip("/")
    if not path:
        return False
    model = db.session.query(Model).filter_by(path=path).first()
    if model is None:
        return False
    db.session.delete(model)
    db.session.commit()
    return True


def tag_get_or_create(name: str, category: str = None) -> Tag:
    """按名称获取或创建标签。"""
    tag = db.session.query(Tag).filter_by(name=name).first()
    if tag is None:
        tag = Tag(name=name, category=category)
        db.session.add(tag)
        db.session.commit()
    return tag


def model_add_tag(model: Model, tag_name: str, category: str = None):
    """为模型添加标签（若不存在则创建标签）。"""
    tag = tag_get_or_create(tag_name, category=category)
    if tag not in model.tags:
        model.tags.append(tag)
    db.session.commit()


def model_remove_tag(model: Model, tag_name: str):
    """移除模型的指定标签。"""
    tag = db.session.query(Tag).filter_by(name=tag_name).first()
    if tag and tag in model.tags:
        model.tags.remove(tag)
        db.session.commit()


# -----------------------------------------------------------------------------
# TestSet
# -----------------------------------------------------------------------------

def testset_upsert(
    testset_id: str,
    name: str,
    hf_dataset: str = None,
    hf_subset: str = None,
    hf_split: str = None,
    lm_eval_task: str = None,
    benchmark_config: dict = None,
    version: str = None,
    sample_count: int = 0,
    is_local: bool = False,
    local_path: str = None,
    yaml_template_path: str = None,
    created_by: str = None,
    notes: str = None,
    question_type: str = None,
    cached_configs: list = None,
    cached_splits: list = None,
    **extra,
) -> TestSet:
    """创建或更新测试集。DB 为主写入源。"""
    row = db.session.get(TestSet, testset_id)
    if row is None:
        row = TestSet(
            id=testset_id,
            name=name,
            type=extra.get("type"),
            hf_dataset=hf_dataset,
            hf_subset=hf_subset,
            hf_split=hf_split,
            lm_eval_task=lm_eval_task,
            benchmark_config=benchmark_config,
            version=version,
            sample_count=sample_count,
            is_local=is_local,
            local_path=local_path,
            yaml_template_path=yaml_template_path,
            created_by=created_by,
            notes=notes,
            question_type=question_type,
            cached_configs=cached_configs,
            cached_splits=cached_splits,
        )
        db.session.add(row)
    else:
        row.name = name
        row.hf_dataset = hf_dataset
        row.hf_subset = hf_subset
        row.hf_split = hf_split
        row.lm_eval_task = lm_eval_task
        row.benchmark_config = benchmark_config
        row.version = version
        row.sample_count = sample_count
        row.is_local = is_local
        row.local_path = local_path
        if extra.get("type") is not None:
            row.type = extra.get("type")
        if yaml_template_path is not None:
            row.yaml_template_path = yaml_template_path
        if created_by is not None:
            row.created_by = created_by
        if notes is not None:
            row.notes = notes
        if question_type is not None:
            row.question_type = question_type
        if cached_configs is not None:
            row.cached_configs = cached_configs
        if cached_splits is not None:
            row.cached_splits = cached_splits
        row.updated_at = datetime.utcnow()
    db.session.commit()
    return row


def testset_update_hf_cache(
    testset_id: str,
    cached_configs: list = None,
    cached_splits: list = None,
) -> bool:
    """更新测试集的 hf_info 缓存字段。"""
    row = db.session.get(TestSet, testset_id)
    if not row:
        return False
    if cached_configs is not None:
        row.cached_configs = cached_configs
    if cached_splits is not None:
        row.cached_splits = cached_splits
    row.updated_at = datetime.utcnow()
    db.session.commit()
    return True


def testset_list_from_db():
    """从 DB 读取测试集列表（以 DB 为主）。"""
    return db.session.query(TestSet).order_by(TestSet.updated_at.desc()).all()


def testset_get_by_id(testset_id: str) -> TestSet | None:
    return db.session.get(TestSet, testset_id)


def testset_enrich_from_eval_result(testset_id: str) -> dict | None:
    """从该测试集下某条 EvaluationResult 取 hf_dataset/hf_subset 等补全 TestSet。返回可合并到 to_dict 的字段，若无则返回 None。"""
    r = (
        db.session.query(EvaluationResult)
        .filter(
            EvaluationResult.testset_id == testset_id,
            EvaluationResult.hf_dataset.isnot(None),
            EvaluationResult.hf_dataset != "",
        )
        .order_by(EvaluationResult.created_at.desc())
        .first()
    )
    if not r:
        return None
    out = {"hf_dataset": r.hf_dataset}
    if r.hf_subset is not None:
        out["hf_subset"] = r.hf_subset
    task = db.session.get(Task, r.task_id)
    if task and task.config and isinstance(task.config, dict) and task.config.get("hf_split"):
        out["hf_split"] = task.config.get("hf_split")
    return out


# -----------------------------------------------------------------------------
# EvaluationResult
# -----------------------------------------------------------------------------

def evaluation_result_get_by_task_id(task_id: str) -> EvaluationResult | None:
    """按 task_id 查询是否已有评估结果（用于迁移脚本去重）。"""
    return db.session.query(EvaluationResult).filter_by(task_id=task_id).first()


def evaluation_result_insert(
    task_id: str,
    testset_id: str,
    accuracy: float = None,
    f1_score: float = None,
    test_cases: int = None,
    context: int = None,
    throughput: float = None,
    time: float = None,
    metrics: dict = None,
    model_id: str = None,
    hf_dataset: str = None,
    hf_subset: str = None,
    efficiency_score: float = None,
) -> EvaluationResult:
    """插入一条评估结果。DB 为主写入源。"""
    row = EvaluationResult(
        task_id=task_id,
        testset_id=testset_id,
        model_id=model_id,
        accuracy=accuracy,
        f1_score=f1_score,
        test_cases=test_cases,
        context=context,
        throughput=throughput,
        time=time,
        metrics=metrics,
        hf_dataset=hf_dataset,
        hf_subset=hf_subset,
        efficiency_score=efficiency_score,
    )
    db.session.add(row)
    db.session.commit()
    return row


# -----------------------------------------------------------------------------
# 只读查询（供「优先 DB、文件回退」使用）
# -----------------------------------------------------------------------------

def task_list_for_fusion_history():
    """融合历史：排除 eval_only，按创建时间倒序。"""
    tasks = (
        db.session.query(Task)
        .filter(Task.task_type != "eval_only")
        .order_by(Task.created_at.desc())
        .all()
    )
    return [t.to_dict() for t in tasks]


def task_list_for_eval_history():
    """评估历史：仅 eval_only，按创建时间倒序。"""
    tasks = (
        db.session.query(Task)
        .filter(Task.task_type == "eval_only")
        .order_by(Task.created_at.desc())
        .all()
    )
    return [t.to_dict() for t in tasks]


def evaluation_results_grouped_by_testset():
    """按 testset_id 分组的评估结果，每条含 model_name（来自 Model）。返回 dict[testset_id, list]。"""
    results = (
        db.session.query(EvaluationResult)
        .order_by(EvaluationResult.created_at.desc())
        .all()
    )
    model_ids = {r.model_id for r in results if r.model_id}
    model_names = {}
    if model_ids:
        for m in db.session.query(Model).filter(Model.id.in_(model_ids)):
            model_names[m.id] = m.name
    by_testset = {}
    for r in results:
        tid = r.testset_id or ""
        if tid not in by_testset:
            by_testset[tid] = []
        created = r.created_at.isoformat() if r.created_at else None
        by_testset[tid].append({
            "task_id": r.task_id,
            "model_name": model_names.get(r.model_id, "Unknown Model"),
            "accuracy": r.accuracy,
            "f1_score": r.f1_score,
            "test_cases": r.test_cases,
            "created_at": created,
            "throughput": r.throughput,
            "time": r.time,
            "context": r.context,
            "hf_dataset": r.hf_dataset,
            "hf_subset": r.hf_subset,
            "efficiency_score": r.efficiency_score,
        })
    return by_testset


def model_list_all():
    """模型表全量列表，按创建时间倒序。"""
    models = db.session.query(Model).order_by(Model.created_at.desc()).all()
    return [m.to_dict() for m in models]


def evaluation_best_per_model_per_testset():
    """按 (testset_id, model_id) 聚合，取每个模型在该测试集上的最佳 accuracy。返回 [(testset_id, model_id, best_acc), ...]。"""
    from sqlalchemy import func
    subq = (
        db.session.query(
            EvaluationResult.testset_id,
            EvaluationResult.model_id,
            func.max(EvaluationResult.accuracy).label("best_acc"),
        )
        .filter(EvaluationResult.model_id.isnot(None))
        .filter(EvaluationResult.testset_id.isnot(None))
        .group_by(EvaluationResult.testset_id, EvaluationResult.model_id)
    ).subquery()
    rows = db.session.query(subq.c.testset_id, subq.c.model_id, subq.c.best_acc).all()
    return [(r[0], r[1], float(r[2]) if r[2] is not None else 0.0) for r in rows]


# -----------------------------------------------------------------------------
# EvolutionStep
# -----------------------------------------------------------------------------

def evolution_steps_bulk_insert(task_id: str, rows: list[dict]):
    """批量插入进化步骤记录（从 CSV 同步时调用）。已有数据则先清除后重写。"""
    db.session.query(EvolutionStep).filter_by(task_id=task_id).delete()
    for r in rows:
        db.session.add(EvolutionStep(
            task_id=task_id,
            eval_index=int(r.get("step") or r.get("eval_index") or 0),
            generation=int(r["generation"]) if r.get("generation") is not None else None,
            accuracy=float(r["objective_1"]) if r.get("objective_1") is not None else float(r.get("accuracy", 0)),
            best_accuracy=float(r["best_acc"]) if r.get("best_acc") is not None else None,
            genotype=[float(r.get("genotype_1", 0)), float(r.get("genotype_2", 0))],
        ))
    db.session.commit()


def evolution_steps_for_task(task_id: str) -> list[dict]:
    """查询指定任务的所有进化步骤（按 eval_index 排序）。"""
    rows = (
        db.session.query(EvolutionStep)
        .filter_by(task_id=task_id)
        .order_by(EvolutionStep.eval_index)
        .all()
    )
    return [r.to_dict() for r in rows]


def evolution_step_update(step_id: int, **kwargs) -> EvolutionStep | None:
    """修改单条进化步骤记录（支持 accuracy / genotype / generation 等字段）。"""
    row = db.session.get(EvolutionStep, step_id)
    if row is None:
        return None
    for k in ("accuracy", "best_accuracy", "genotype", "generation"):
        if k in kwargs:
            setattr(row, k, kwargs[k])
    row.updated_at = datetime.utcnow() if hasattr(row, "updated_at") else None
    db.session.commit()
    return row
