"""Phase 0 数据加载：从 ALFRED Lite 轨迹 JSON 中筛出"厨房 + 2 个任务类型"的题目，
蒸馏出 Pilot 需要的字段（含 high_pddl 子目标序列 = GT 派生原料），写成小 jsonl。

用法（在 WSL 中运行，需先下载 ALFRED，详见 README）：
    python embodied_skill/data/code/load_tasks.py

输入：~/alfred_data/json_2.1.0/{train,valid_seen,valid_unseen}/.../traj_data.json
输出：embodied_skill/data/datasets/tasks_kitchen.jsonl        (全部命中题)
      embodied_skill/data/datasets/tasks_kitchen_sample.jsonl (均衡抽样 ~100 题)
"""
import os
import json
import glob
import random
from collections import Counter, defaultdict

ALFRED_ROOT = os.path.expanduser("~/alfred_data/json_2.1.0")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
# WP1（Stage II）：任务类型 2→4，全部落在"效果签名 GT"可信支持范围内。
#   pick_and_place_simple          -> at_receptacle
#   pick_heat_then_place_in_recep  -> isHot + at_receptacle
#   pick_cool_then_place_in_recep  -> isCold + at_receptacle
#   pick_clean_then_place_in_recep -> isClean + at_receptacle
# （slice 作为修饰：pddl_params.object_sliced 触发 isSliced，可叠加在上述类型上）
# 明确排除（GT 框架不覆盖，见 stage2 计划 WP1）：
#   pick_two_obj_and_place          -> 需物体实例识别/计数（LoTa-Bench 亦排除）
#   pick_and_place_with_movable_recep -> "放进可移动容器再整体放置"双重放置，at_receptacle 无法如实刻画
#   look_at_obj_in_light            -> examine 无对应效果 token（且厨房场景本就无此类）
WANT_TYPES = {
    "pick_and_place_simple",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
}
# AI2-THOR 厨房场景编号为 1~30
def is_kitchen(scene_num):
    return scene_num is not None and 1 <= int(scene_num) <= 30

SAMPLE_PER_TYPE = 50  # 每类抽多少（用于快速 sample 集）


def load_one(traj_path):
    try:
        with open(traj_path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        return None
    tt = d.get("task_type")
    scene = d.get("scene", {}).get("scene_num")
    if tt not in WANT_TYPES or not is_kitchen(scene):
        return None
    high = []
    for h in d.get("plan", {}).get("high_pddl", []):
        act = h.get("discrete_action", {})
        high.append({"action": act.get("action"), "args": act.get("args", [])})
    anns = d.get("turk_annotations", {}).get("anns", [])
    instruction = anns[0].get("task_desc", "").strip() if anns else ""
    step_instr = anns[0].get("high_descs", []) if anns else []
    return {
        "task_id": d.get("task_id"),
        "task_type": tt,
        "scene_num": int(scene),
        "floor_plan": d.get("scene", {}).get("floor_plan"),
        "pddl_params": d.get("pddl_params", {}),
        "high_pddl_actions": high,     # 子目标动作序列（GT 原料）
        "instruction": instruction,     # 任务级自然语言（router 的 query）
        "step_instructions": step_instr,
        "source_path": traj_path,
    }


def main():
    if not os.path.isdir(ALFRED_ROOT):
        raise SystemExit(f"未找到数据目录 {ALFRED_ROOT}，请先运行 fetch_alfred_lite.sh 下载解压。")
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    for split in ["train", "valid_seen", "valid_unseen"]:
        pattern = os.path.join(ALFRED_ROOT, split, "*", "*", "traj_data.json")
        for tj in glob.glob(pattern):
            r = load_one(tj)
            if r:
                r["split"] = split
                rows.append(r)

    # 统计
    by_type = Counter(r["task_type"] for r in rows)
    by_split = Counter(r["split"] for r in rows)
    print(f"== 命中题目统计（厨房 + {len(WANT_TYPES)} 类型） ==")
    print("总数:", len(rows))
    print("按类型:", dict(by_type))
    print("按 split:", dict(by_split))

    def dump(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    # 全量写出（含 split 字段）
    full_path = dump(os.path.join(OUT_DIR, "tasks_kitchen.jsonl"), rows)
    print("已写全量:", full_path, f"({len(rows)} 题)")

    # WP9(a)：headline 评测集 = valid_seen + valid_unseen（按 ALFRED 官方 split，天然泛化轴）
    eval_rows = [r for r in rows if r["split"] in ("valid_seen", "valid_unseen")]
    eval_path = dump(os.path.join(OUT_DIR, "tasks_kitchen_eval.jsonl"), eval_rows)
    ev_split = Counter(r["split"] for r in eval_rows)
    ev_type = Counter(r["task_type"] for r in eval_rows)
    print(f"已写评测集(valid_seen+unseen, {len(eval_rows)} 题):", eval_path)
    print("  eval 按 split:", dict(ev_split), "| 按类型:", dict(ev_type))

    # WP5：训练池 = train split（供 L0~L4 训练/合成，与评测场景解耦）
    train_rows = [r for r in rows if r["split"] == "train"]
    train_path = dump(os.path.join(OUT_DIR, "tasks_kitchen_train.jsonl"), train_rows)
    print(f"已写训练集(train, {len(train_rows)} 题):", train_path)

    # 均衡抽样（每类各取 SAMPLE_PER_TYPE，快速冒烟用；优先 valid 便于当评测）
    random.seed(42)
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["task_type"]].append(r)
    sample = []
    for tt, items in buckets.items():
        random.shuffle(items)
        sample.extend(items[:SAMPLE_PER_TYPE])
    sample_path = dump(os.path.join(OUT_DIR, "tasks_kitchen_sample.jsonl"), sample)
    print(f"已写抽样({len(sample)} 题):", sample_path)

    # 打印 2 个示例，直观看 high_pddl 子目标序列
    print("\n== 示例（每类 1 个） ==")
    shown = set()
    for r in rows:
        if r["task_type"] in shown:
            continue
        shown.add(r["task_type"])
        seq = " -> ".join(a["action"] for a in r["high_pddl_actions"])
        print(f"\n[{r['task_type']}] scene={r['floor_plan']} id={r['task_id']}")
        print("  指令:", r["instruction"][:90])
        print("  子目标序列:", seq)
        print("  pddl_params:", r["pddl_params"])
        if len(shown) >= len(WANT_TYPES):
            break


if __name__ == "__main__":
    main()
