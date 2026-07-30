"""原子技能的可执行实现（真实调用 AI2-THOR）。

约定：
- 每个技能签名 `fn(ctrl, ...) -> (event, ok: bool, info: str)`。
- 物体用"类型名"（如 "Apple"/"Microwave"，大小写/符号不敏感）指定，函数内部解析为最近的 objectId。
- 导航简化为 GetInteractablePoses + Teleport（聚焦"选技能"，不学底层导航）。
- 交互统一 forceAction=True，减少可见性/距离带来的偶发失败（Pilot 关注路由，不苛求执行细节）。

用于 Phase 3 执行器；Phase 1 用 smoke_atomic.py 做执行冒烟验证。
"""
import re


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _dist(a, b):
    return ((a["x"] - b["x"]) ** 2 + (a["z"] - b["z"]) ** 2) ** 0.5


# 领域别名：某些技能用通称指代一类物体
_ALIASES = {
    "knife": {"knife", "butterknife"},  # 切片用任意刀具
}

# ALFRED 标注名 -> AI2-THOR objectType（归一化后的键）。
# 由 exec/probe_vocab.py 数据驱动得出：厨房 2 类任务里唯一对不上的是 Glassbottle。
# 若将来扩任务类型出现新差异，重跑 probe_vocab.py 即可发现并在此补一条。
_VOCAB_MAP = {
    "glassbottle": "bottle",
}


def find_objects(event, type_or_id):
    """按 objectId 精确匹配，否则按 objectType 归一化匹配（含词表映射 + 别名 + 切片变体）。

    - 词表映射：ALFRED 名（如 Glassbottle）→ THOR objectType（Bottle）。
    - 切片变体：查询 "Apple" 时也接受 "AppleSliced"（AI2-THOR 切片后原物体被替换为
      <Type>Sliced 的切片对象）。这样切片子技能之后仍能导航/拿起/放置切片。
    """
    objs = event.metadata["objects"]
    for o in objs:
        if o["objectId"] == type_or_id:
            return [o]
    key = _norm(type_or_id)
    key = _VOCAB_MAP.get(key, key)          # 词表映射
    accept = set(_ALIASES.get(key, {key}))
    accept.add(key + "sliced")              # 切片变体
    return [o for o in objs if _norm(o["objectType"]) in accept]


def _resolve_nearest(ctrl, type_or_id):
    ev = ctrl.last_event
    cands = find_objects(ev, type_or_id)
    if not cands:
        return None
    ap = ev.metadata["agent"]["position"]
    return min(cands, key=lambda o: _dist(ap, o["position"]))


def _held(event):
    held = [o for o in event.metadata["objects"] if o.get("isPickedUp")]
    return held[0] if held else None


# ---------- 原子技能实现 ----------

def goto_object(ctrl, target):
    ev = ctrl.last_event
    cands = find_objects(ev, target)
    if not cands:
        return ev, False, f"no object of type {target}"
    ap = ev.metadata["agent"]["position"]
    cands.sort(key=lambda o: _dist(ap, o["position"]))
    # 在同类型候选里挑第一个"够得到(有可交互姿态)"的，而非单纯最近的
    last_err = "no candidate reachable"
    for obj in cands:
        oid = obj["objectId"]
        poses = ctrl.step("GetInteractablePoses", objectId=oid).metadata["actionReturn"]
        if not poses:
            last_err = f"no interactable pose for {oid}"
            continue
        p = poses[0]
        ev = ctrl.step(
            "Teleport", x=p["x"], y=p["y"], z=p["z"],
            rotation=p["rotation"], horizon=p["horizon"], standing=p["standing"],
        )
        if ev.metadata["lastActionSuccess"]:
            return ev, True, oid
        last_err = f"teleport failed for {oid}"
    return ctrl.last_event, False, last_err


def pickup_object(ctrl, obj):
    ev0 = ctrl.last_event
    cands = find_objects(ev0, obj)
    if not cands:
        return ev0, False, f"no object {obj}"
    ap = ev0.metadata["agent"]["position"]
    # 优先可拾取候选（切片后原整只 isSliced 但 pickupable=False，须选切片）
    pick = [o for o in cands if o.get("pickupable")] or cands
    pick.sort(key=lambda o: _dist(ap, o["position"]))
    o = pick[0]
    ev = ctrl.step("PickupObject", objectId=o["objectId"], forceAction=True)
    return ev, ev.metadata["lastActionSuccess"], ev.metadata.get("errorMessage", "")


def _robust_put(ctrl, rid, held, tries_coords=25):
    """把手持物放进/放到容器 rid，尽最大努力：
    ① PutObject(placeStationary True→False)；
    ② 失败则回退 GetSpawnCoordinatesAboveReceptacle 取容器内候选点 + PlaceObjectAtPoint
       强制落点（解决抽屉/水槽/冰箱"No valid positions to place object found"长尾）。
    返回 (event, ok, info)。held 需为放置前手持物体的 metadata（含 objectId）。"""
    last_err = ""
    for stationary in (True, False):
        ev = ctrl.step("PutObject", objectId=rid, forceAction=True, placeStationary=stationary)
        if ev.metadata["lastActionSuccess"]:
            return ev, True, rid
        last_err = ev.metadata.get("errorMessage", "") or last_err
    # 回退：容器上方/内部候选点逐个 PlaceObjectAtPoint
    if held is not None:
        sp = ctrl.step("GetSpawnCoordinatesAboveReceptacle", objectId=rid, anywhere=True)
        coords = sp.metadata.get("actionReturn") or []
        ap = ctrl.last_event.metadata["agent"]["position"]
        coords.sort(key=lambda p: (p["x"] - ap["x"]) ** 2 + (p["z"] - ap["z"]) ** 2)
        for p in coords[:tries_coords]:
            ev = ctrl.step("PlaceObjectAtPoint", objectId=held["objectId"], position=p)
            if ev.metadata["lastActionSuccess"]:
                return ev, True, rid + "@point"
            last_err = ev.metadata.get("errorMessage", "") or last_err
    return ctrl.last_event, False, last_err or "no valid put position"


def put_object(ctrl, obj, receptacle):
    """把手持物放到目标容器。多落点重试：遍历同类型最近的若干容器实例，
    每个先试 PutObject（稳定→下落），失败再回退坐标落点（_robust_put）。"""
    ev0 = ctrl.last_event
    held = _held(ev0)
    cands = find_objects(ev0, receptacle)
    if not cands:
        return ev0, False, f"no receptacle {receptacle}"
    ap = ev0.metadata["agent"]["position"]
    cands.sort(key=lambda o: _dist(ap, o["position"]))
    last_err = "no candidate placeable"
    for r in cands[:4]:                      # 最多试最近的 4 个同类容器实例
        rid = r["objectId"]
        # 可开合容器（冰箱/微波炉/柜子）若关着，先开门再放，放完关门
        opened = False
        if r.get("openable") and not r.get("isOpen"):
            ctrl.step("OpenObject", objectId=rid, forceAction=True)
            opened = True
        ev, ok, info = _robust_put(ctrl, rid, held)
        if not ok:
            last_err = info or last_err
        if opened:
            ctrl.step("CloseObject", objectId=rid, forceAction=True)
        if ok:
            return ev, True, info
    return ctrl.last_event, False, last_err


def open_receptacle(ctrl, receptacle):
    r = _resolve_nearest(ctrl, receptacle)
    if r is None:
        return ctrl.last_event, False, f"no receptacle {receptacle}"
    ev = ctrl.step("OpenObject", objectId=r["objectId"], forceAction=True)
    return ev, ev.metadata["lastActionSuccess"], ev.metadata.get("errorMessage", "")


def close_receptacle(ctrl, receptacle):
    r = _resolve_nearest(ctrl, receptacle)
    if r is None:
        return ctrl.last_event, False, f"no receptacle {receptacle}"
    ev = ctrl.step("CloseObject", objectId=r["objectId"], forceAction=True)
    return ev, ev.metadata["lastActionSuccess"], ev.metadata.get("errorMessage", "")


def toggle_on(ctrl, obj):
    o = _resolve_nearest(ctrl, obj)
    if o is None:
        return ctrl.last_event, False, f"no object {obj}"
    ev = ctrl.step("ToggleObjectOn", objectId=o["objectId"], forceAction=True)
    return ev, ev.metadata["lastActionSuccess"], ev.metadata.get("errorMessage", "")


def toggle_off(ctrl, obj):
    o = _resolve_nearest(ctrl, obj)
    if o is None:
        return ctrl.last_event, False, f"no object {obj}"
    ev = ctrl.step("ToggleObjectOff", objectId=o["objectId"], forceAction=True)
    return ev, ev.metadata["lastActionSuccess"], ev.metadata.get("errorMessage", "")


def slice_object(ctrl, obj):
    o = _resolve_nearest(ctrl, obj)
    if o is None:
        return ctrl.last_event, False, f"no object {obj}"
    ev = ctrl.step("SliceObject", objectId=o["objectId"], forceAction=True)
    return ev, ev.metadata["lastActionSuccess"], ev.metadata.get("errorMessage", "")


def fill_with_liquid(ctrl, obj, liquid="water"):
    o = _resolve_nearest(ctrl, obj)
    if o is None:
        return ctrl.last_event, False, f"no object {obj}"
    ev = ctrl.step("FillObjectWithLiquid", objectId=o["objectId"], fillLiquid=liquid, forceAction=True)
    return ev, ev.metadata["lastActionSuccess"], ev.metadata.get("errorMessage", "")


def empty_liquid(ctrl, obj):
    o = _resolve_nearest(ctrl, obj)
    if o is None:
        return ctrl.last_event, False, f"no object {obj}"
    ev = ctrl.step("EmptyLiquidFromObject", objectId=o["objectId"], forceAction=True)
    return ev, ev.metadata["lastActionSuccess"], ev.metadata.get("errorMessage", "")


# ---------- 复合型原子技能（body 是原语序列，但在路由层视为一个子目标） ----------

def _appliance_cycle(ctrl, appliance_type, obj, toggle=True):
    """通用：开门→(把手持物放入)→关门→[开关电器]→开门→取回物体→关门。
    用于微波炉加热 / 冰箱冷却。返回 (event, ok, info)。"""
    app = _resolve_nearest(ctrl, appliance_type)
    if app is None:
        return ctrl.last_event, False, f"no {appliance_type}"
    aid = app["objectId"]
    held = _held(ctrl.last_event)
    if held is None:
        return ctrl.last_event, False, "hand empty; must hold object first"
    held_type = held["objectType"]

    ctrl.step("OpenObject", objectId=aid, forceAction=True)
    ev, put_ok, put_info = _robust_put(ctrl, aid, held)
    if not put_ok:
        return ev, False, "put into appliance failed: " + put_info
    ctrl.step("CloseObject", objectId=aid, forceAction=True)
    if toggle:
        ctrl.step("ToggleObjectOn", objectId=aid, forceAction=True)
        ctrl.step("ToggleObjectOff", objectId=aid, forceAction=True)
    ctrl.step("OpenObject", objectId=aid, forceAction=True)
    # 取回物体（在电器里重新解析同类型物体）
    back = _resolve_nearest(ctrl, held_type)
    if back is None:
        return ctrl.last_event, False, f"cannot find {held_type} to pick back"
    ev = ctrl.step("PickupObject", objectId=back["objectId"], forceAction=True)
    ctrl.step("CloseObject", objectId=aid, forceAction=True)
    return ev, ev.metadata["lastActionSuccess"], back["objectId"]


def heat_object_microwave(ctrl, obj):
    ev, ok, info = _appliance_cycle(ctrl, "Microwave", obj, toggle=True)
    if not ok:
        return ev, False, info
    o = _resolve_nearest(ctrl, obj)
    is_hot = (o or {}).get("temperature") == "Hot"
    return ev, is_hot, f"temperature={(o or {}).get('temperature')}"


def cool_object_fridge(ctrl, obj):
    ev, ok, info = _appliance_cycle(ctrl, "Fridge", obj, toggle=False)
    if not ok:
        return ev, False, info
    o = _resolve_nearest(ctrl, obj)
    is_cold = (o or {}).get("temperature") == "Cold"
    return ev, is_cold, f"temperature={(o or {}).get('temperature')}"


def clean_object_sink(ctrl, obj):
    """放入水槽→开水龙头→关水龙头→取回，之后物体变干净。"""
    sink = _resolve_nearest(ctrl, "SinkBasin")
    if sink is None:
        return ctrl.last_event, False, "no SinkBasin"
    held = _held(ctrl.last_event)
    if held is None:
        return ctrl.last_event, False, "hand empty"
    held_type = held["objectType"]
    ev, put_ok, _ = _robust_put(ctrl, sink["objectId"], held)
    if not put_ok:
        return ev, False, "put into sink failed"
    faucet = _resolve_nearest(ctrl, "Faucet")
    if faucet is not None:
        ctrl.step("ToggleObjectOn", objectId=faucet["objectId"], forceAction=True)
        ctrl.step("ToggleObjectOff", objectId=faucet["objectId"], forceAction=True)
    back = _resolve_nearest(ctrl, held_type)
    if back is None:
        return ctrl.last_event, False, f"cannot find {held_type} to pick back"
    ev = ctrl.step("PickupObject", objectId=back["objectId"], forceAction=True)
    o = _resolve_nearest(ctrl, held_type)
    is_clean = not (o or {}).get("isDirty", False)
    return ev, is_clean, f"isDirty={(o or {}).get('isDirty')}"


# 技能名 -> 函数 的分发表（body 字段 "atomic_bodies.xxx" 用得到）
REGISTRY = {
    "goto_object": goto_object,
    "pickup_object": pickup_object,
    "put_object": put_object,
    "open_receptacle": open_receptacle,
    "close_receptacle": close_receptacle,
    "toggle_on": toggle_on,
    "toggle_off": toggle_off,
    "slice_object": slice_object,
    "fill_with_liquid": fill_with_liquid,
    "empty_liquid": empty_liquid,
    "heat_object_microwave": heat_object_microwave,
    "cool_object_fridge": cool_object_fridge,
    "clean_object_sink": clean_object_sink,
}
