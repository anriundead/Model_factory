#!/usr/bin/env python3
"""
数据库接入与方案 A（A1 模型删除同步 DB、A2 优先读 DB + 文件回退）完整验证流程。

运行方式（在项目根目录 mergeKit_beta 下）:
  python scripts/verify_db_integration.py

或:
  cd /path/to/mergeKit_beta && python scripts/verify_db_integration.py
"""
from __future__ import print_function

import os
import sys

# 保证项目根在 path 中
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.chdir(_ROOT)


def run():
    passed = 0
    failed = 0

    # -------------------------------------------------------------------------
    # 1. 应用启动与 DB 连接
    # -------------------------------------------------------------------------
    print("\n--- 1. 应用启动与 DB 连接 ---")
    try:
        from app import create_app
        app = create_app()
        with app.app_context():
            from app.extensions import db
            from app.models import Task, Model, EvaluationResult, TestSet
            _ = db.session.query(Task).first()
            _ = db.session.query(Model).first()
        passed += 1
        print("[PASS] 应用启动成功，DB 表可访问")
    except Exception as e:
        failed += 1
        print("[FAIL] 应用启动:", e)
        if "create_app" not in str(e):
            import traceback
            traceback.print_exc()
        return passed, failed

    # -------------------------------------------------------------------------
    # 2. A2 DB 读层：融合历史 / 评估历史 / 排行榜 / 模型列表
    # -------------------------------------------------------------------------
    print("\n--- 2. A2 DB 读层（db_read_layer）---")
    with app.app_context():
        try:
            from app.db_read_layer import (
                get_fusion_history_from_db,
                get_eval_history_from_db,
                get_leaderboard_from_db,
                get_model_repo_list_from_db,
            )
            h = get_fusion_history_from_db()
            assert h is None or isinstance(h, list), "fusion_history 应为 list 或 None"
            e = get_eval_history_from_db()
            assert e is None or isinstance(e, list), "eval_history 应为 list 或 None"
            lb = get_leaderboard_from_db()
            assert lb is None or isinstance(lb, dict), "leaderboard 应为 dict 或 None"
            # 模型列表需要 merge_metadata 函数，用 None 即可（仅测不报错）
            ml = get_model_repo_list_from_db(lambda p: None)
            assert ml is None or isinstance(ml, list), "model_list 应为 list 或 None"
            passed += 1
            print("[PASS] db_read_layer 四类接口返回类型正确 (fusion=%s eval=%s lb=%s model=%s)" % (
                len(h) if h else 0, len(e) if e else 0, len(lb) if lb else 0, len(ml) if ml else 0,
            ))
        except Exception as ex:
            failed += 1
            print("[FAIL] db_read_layer:", ex)
            import traceback
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # 3. A2 Services：get_all_history / get_all_eval_history / load_leaderboard / model_repo_list
    # -------------------------------------------------------------------------
    print("\n--- 3. A2 Services 优先 DB + 文件回退 ---")
    try:
        state = getattr(app, "state", None) or getattr(app, "_state", None)
        if state is None:
            from app.state import AppState
            from config import Config
            Config.setup_environment()
            state = AppState()
        from app.services import Services
        services = Services(state)
        services.app = app

        hist = services.get_all_history()
        assert isinstance(hist, list), "get_all_history 应返回 list"
        eval_hist = services.get_all_eval_history()
        assert isinstance(eval_hist, list), "get_all_eval_history 应返回 list"
        leaderboards = services.load_leaderboard()
        assert isinstance(leaderboards, dict), "load_leaderboard 应返回 dict"
        repo_list = services.model_repo_list()
        assert isinstance(repo_list, list), "model_repo_list 应返回 list"

        passed += 1
        print("[PASS] get_all_history=%d, get_all_eval_history=%d, load_leaderboard keys=%d, model_repo_list=%d" % (
            len(hist), len(eval_hist), len(leaderboards), len(repo_list),
        ))
    except Exception as ex:
        failed += 1
        print("[FAIL] Services 读接口:", ex)
        import traceback
        traceback.print_exc()

    # -------------------------------------------------------------------------
    # 4. A1 模型删除同步 DB：注册 → 删除 → 校验 DB 中已无该 path
    # -------------------------------------------------------------------------
    print("\n--- 4. A1 模型删除同步 DB ---")
    test_path = "/tmp/mergekit_verify_test_model_path_do_not_use"
    with app.app_context():
        try:
            from app.repositories import model_register, model_get_by_path, model_delete_by_path
            model_register(path=test_path, name="VerifyTestModel", source="merged", task_id="verify_task")
            m = model_get_by_path(test_path)
            assert m is not None, "注册后应能按 path 查到"
            deleted = model_delete_by_path(test_path)
            assert deleted is True, "删除应返回 True"
            m2 = model_get_by_path(test_path)
            assert m2 is None, "删除后应查不到"
            passed += 1
            print("[PASS] model_register → model_delete_by_path → 记录已从 DB 移除")
        except Exception as ex:
            failed += 1
            print("[FAIL] A1 模型删除同步:", ex)
            import traceback
            traceback.print_exc()
            # 尝试清理
            try:
                with app.app_context():
                    from app.repositories import model_delete_by_path
                    model_delete_by_path(test_path)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # 5. API 层：GET /api/history, /api/test_history, /api/testset/list
    # -------------------------------------------------------------------------
    print("\n--- 5. API 层（HTTP）---")
    try:
        client = app.test_client()
        r = client.get("/api/history")
        assert r.status_code == 200, "GET /api/history 应为 200"
        data = r.get_json()
        assert data.get("status") == "success" and "history" in data

        r2 = client.get("/api/test_history")
        assert r2.status_code == 200, "GET /api/test_history 应为 200"
        data2 = r2.get_json()
        assert data2.get("status") == "success" and "history" in data2

        r3 = client.get("/api/testset/list")
        assert r3.status_code == 200, "GET /api/testset/list 应为 200"
        data3 = r3.get_json()
        assert data3.get("status") == "success" and "testsets" in data3

        passed += 1
        print("[PASS] GET /api/history, /api/test_history, /api/testset/list 返回 200 且结构正确")
    except Exception as ex:
        failed += 1
        print("[FAIL] API 层:", ex)
        import traceback
        traceback.print_exc()

    # -------------------------------------------------------------------------
    # 6. 模型仓库相关 API（依赖 model_repo_list）
    # -------------------------------------------------------------------------
    print("\n--- 6. 模型仓库 API ---")
    try:
        client = app.test_client()
        # 可能的路由：/api/merged_models 或模型仓库列表接口
        r = client.get("/api/merged_models")
        assert r.status_code == 200, "GET /api/merged_models 应为 200"
        data = r.get_json()
        assert data.get("status") == "success" and "models" in data
        passed += 1
        print("[PASS] GET /api/merged_models 返回 200，models 为 list (len=%d)" % len(data.get("models", [])))
    except Exception as ex:
        failed += 1
        print("[FAIL] 模型仓库 API:", ex)
        import traceback
        traceback.print_exc()

    # -------------------------------------------------------------------------
    # 7. 文件回退：无 DB 数据时仍能从文件返回（不报错）
    # -------------------------------------------------------------------------
    print("\n--- 7. 文件回退（行为保持）---")
    try:
        hist = services.get_all_history()
        eval_hist = services.get_all_eval_history()
        leaderboards = services.load_leaderboard()
        assert isinstance(hist, list) and isinstance(eval_hist, list) and isinstance(leaderboards, dict)
        passed += 1
        print("[PASS] 再次调用 history/leaderboard 不报错，类型正确")
    except Exception as ex:
        failed += 1
        print("[FAIL] 文件回退:", ex)
        import traceback
        traceback.print_exc()

    # -------------------------------------------------------------------------
    # 8. 进化任务进度机制（progress.json → read_evolution_progress → percent/message）
    # -------------------------------------------------------------------------
    print("\n--- 8. 进化任务进度机制 ---")
    fake_task_id = "verify_evo_progress"
    merge_dir = getattr(state, "merge_dir", os.path.join(_ROOT, "merges"))
    task_dir = os.path.join(merge_dir, fake_task_id)
    progress_path = os.path.join(task_dir, "progress.json")
    try:
        os.makedirs(task_dir, exist_ok=True)
        with open(progress_path, "w", encoding="utf-8") as f:
            import json as _json
            _json.dump({
                "status": "running",
                "percent": 42,
                "message": "Step 17/40 (acc=0.55)",
                "current_step": 17,
                "total_expected_steps": 40,
            }, f, ensure_ascii=False, indent=2)
        evo = services.read_evolution_progress(fake_task_id)
        assert evo is not None, "read_evolution_progress 应返回数据"
        assert evo.get("percent") == 42, "percent 应从 progress.json 读取"
        assert "17" in (evo.get("message") or ""), "message 应从 progress.json 读取"
        passed += 1
        print("[PASS] read_evolution_progress 返回 percent=42、message 含 Step，进度机制正常")
    except Exception as ex:
        failed += 1
        print("[FAIL] 进化任务进度机制:", ex)
        import traceback
        traceback.print_exc()
    finally:
        try:
            if os.path.isfile(progress_path):
                os.remove(progress_path)
            if os.path.isdir(task_dir):
                os.rmdir(task_dir)
        except Exception:
            pass

    return passed, failed


def main():
    print("=" * 60)
    print("数据库接入与方案 A 完整验证流程")
    print("=" * 60)
    passed, failed = run()
    print("\n" + "=" * 60)
    print("合计: PASS=%d, FAIL=%d" % (passed, failed))
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
