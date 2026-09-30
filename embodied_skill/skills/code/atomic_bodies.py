"""Executable implementations of the atomic skills (real AI2-THOR calls).

Conventions:
- Every skill has signature `fn(ctrl, ...) -> (event, ok: bool, info: str)`.
- Objects are specified by "type name" (e.g. "Apple"/"Microwave", case/symbol
  insensitive); the function resolves this to the nearest objectId internally.
- Navigation is simplified to GetInteractablePoses + Teleport (we focus on
  "skill selection", not learning low-level navigation).
- Interactions uniformly use forceAction=True to reduce sporadic failures from
  visibility/distance (the pilot focuses on routing, not execution minutiae).

Used by the execution harness (exec/) to run selected skills in AI2-THOR.
"""
import re


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _dist(a, b):
    return ((a["x"] - b["x"]) ** 2 + (a["z"] - b["z"]) ** 2) ** 0.5


# Domain aliases: some skills use a generic term to refer to a class of objects
_ALIASES = {
    "knife": {"knife", "butterknife"},  # any knife can be used for slicing
}

# ALFRED annotation name -> AI2-THOR objectType (normalized keys).
# Derived data-driven from exec/probe_vocab.py: across the 2 kitchen task types
# the only mismatch is Glassbottle.
# If new task types introduce new discrepancies, rerun probe_vocab.py to find
# them and add an entry here.
_VOCAB_MAP = {
    "glassbottle": "bottle",
}


def find_objects(event, type_or_id):
    """Exact match by objectId, otherwise normalized match by objectType
    (including vocab mapping + aliases + sliced variants).

    - Vocab mapping: ALFRED name (e.g. Glassbottle) -> THOR objectType (Bottle).
    - Sliced variant: a query for "Apple" also accepts "AppleSliced" (after
      slicing, AI2-THOR replaces the original object with a <Type>Sliced object).
      This lets slice sub-skills still navigate to / pick up / place the slice.
    """
    objs = event.metadata["objects"]
    for o in objs:
        if o["objectId"] == type_or_id:
            return [o]
    key = _norm(type_or_id)
    key = _VOCAB_MAP.get(key, key)          # vocab mapping
    accept = set(_ALIASES.get(key, {key}))
    accept.add(key + "sliced")              # sliced variant
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


# ---------- Atomic skill implementations ----------

def goto_object(ctrl, target):
    ev = ctrl.last_event
    cands = find_objects(ev, target)
    if not cands:
        return ev, False, f"no object of type {target}"
    ap = ev.metadata["agent"]["position"]
    cands.sort(key=lambda o: _dist(ap, o["position"]))
    # Among same-type candidates, pick the first "reachable" one (has an
    # interactable pose), not simply the nearest.
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
    # Prefer pickupable candidates (after slicing the original whole object is
    # isSliced but pickupable=False, so we must pick the slice).
    pick = [o for o in cands if o.get("pickupable")] or cands
    pick.sort(key=lambda o: _dist(ap, o["position"]))
    o = pick[0]
    ev = ctrl.step("PickupObject", objectId=o["objectId"], forceAction=True)
    return ev, ev.metadata["lastActionSuccess"], ev.metadata.get("errorMessage", "")


def _robust_put(ctrl, rid, held, tries_coords=25):
    """Place the held object into/onto container rid, best-effort:
    (1) PutObject(placeStationary True -> False);
    (2) on failure, fall back to GetSpawnCoordinatesAboveReceptacle to get
        candidate points inside the container + PlaceObjectAtPoint to force
        placement (handles the drawer/sink/fridge "No valid positions to place
        object found" long tail).
    Returns (event, ok, info). `held` must be the metadata (with objectId) of the
    held object before placement."""
    last_err = ""
    for stationary in (True, False):
        ev = ctrl.step("PutObject", objectId=rid, forceAction=True, placeStationary=stationary)
        if ev.metadata["lastActionSuccess"]:
            return ev, True, rid
        last_err = ev.metadata.get("errorMessage", "") or last_err
    # Fallback: try candidate points above/inside the container one by one via PlaceObjectAtPoint
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
    """Place the held object onto the target container. Multi-point retry:
    iterate over the nearest few container instances of the same type; for each,
    first try PutObject (stationary -> drop), then fall back to coordinate-based
    placement (_robust_put)."""
    ev0 = ctrl.last_event
    held = _held(ev0)
    cands = find_objects(ev0, receptacle)
    if not cands:
        return ev0, False, f"no receptacle {receptacle}"
    ap = ev0.metadata["agent"]["position"]
    cands.sort(key=lambda o: _dist(ap, o["position"]))
    last_err = "no candidate placeable"
    for r in cands[:4]:                      # try at most the nearest 4 container instances of this type
        rid = r["objectId"]
        # For openable containers (fridge/microwave/cabinet) that are closed,
        # open first, place, then close.
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


def _contained_in(ctrl, aid, obj_type):
    """Is an object of obj_type actually inside the appliance aid? AI2-THOR reports success for
    a forced placement that in fact lands the object beside the appliance, so containment has to
    be re-checked through receptacleObjectIds rather than trusted."""
    ev = ctrl.last_event
    app = next((o for o in ev.metadata["objects"] if o["objectId"] == aid), None)
    inside = set((app or {}).get("receptacleObjectIds") or [])
    for o in find_objects(ev, obj_type):
        if o["objectId"] in inside:
            return True, o
    return False, None


def _ensure_contained(ctrl, aid, obj_type, tries_coords=25):
    """Assert the object is inside the appliance before the appliance is cycled; if not, force it
    in through GetSpawnCoordinatesAboveReceptacle + PlaceObjectAtPoint and re-check. Without this
    the appliance would be toggled on an empty cavity and the effect would be lost silently."""
    ok, _ = _contained_in(ctrl, aid, obj_type)
    if ok:
        return True, "already-inside"
    target = _resolve_nearest(ctrl, obj_type)
    if target is None:
        return False, f"cannot locate {obj_type} to re-place"
    sp = ctrl.step("GetSpawnCoordinatesAboveReceptacle", objectId=aid, anywhere=True)
    coords = sp.metadata.get("actionReturn") or []
    app = next((o for o in ctrl.last_event.metadata["objects"] if o["objectId"] == aid), None)
    if app is not None:
        ac = app["position"]
        coords.sort(key=lambda p: (p["x"] - ac["x"]) ** 2 + (p["z"] - ac["z"]) ** 2)
    for p in coords[:tries_coords]:
        ctrl.step("PlaceObjectAtPoint", objectId=target["objectId"], position=p)
        ok, _ = _contained_in(ctrl, aid, obj_type)
        if ok:
            return True, "re-placed-inside"
    return False, "object not inside appliance after re-place"


# ---------- Composite atomic skills (body is a primitive sequence, but treated as one sub-goal at the routing layer) ----------

def _appliance_cycle(ctrl, appliance_type, obj, toggle=True):
    """Generic: open -> (put held object in) -> close -> [toggle appliance] ->
    open -> pick object back -> close. Used for microwave heating / fridge
    cooling. Returns (event, ok, info)."""
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
    in_ok, in_info = _ensure_contained(ctrl, aid, held_type)
    if not in_ok:
        return ctrl.last_event, False, f"not contained in {appliance_type}: {in_info}"
    ctrl.step("CloseObject", objectId=aid, forceAction=True)
    if toggle:
        ctrl.step("ToggleObjectOn", objectId=aid, forceAction=True)
        ctrl.step("ToggleObjectOff", objectId=aid, forceAction=True)
    ctrl.step("OpenObject", objectId=aid, forceAction=True)
    # Pick the object back (re-resolve the same-type object inside the appliance)
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
    """Put into sink -> turn faucet on -> turn faucet off -> pick back; the
    object becomes clean afterwards."""
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
    in_ok, in_info = _ensure_contained(ctrl, sink["objectId"], held_type)
    if not in_ok:
        return ctrl.last_event, False, f"not contained in SinkBasin: {in_info}"
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


# Skill name -> function dispatch table (used by the body field "atomic_bodies.xxx")
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
