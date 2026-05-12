# FILE: banana_triple_view_node.py
from __future__ import annotations

import copy
import json
import os
import re
import sys
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np
from PIL import Image

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

try:
    from banana_node import (
        logger,
        _MANIFEST,
        _NODE,
        _cfg,
        _cfg_or_manifest,
        _enum_source_options,
        _enum_source_display,
        _manual_model_default,
        _manual_image_size_default,
        _manual_aspect_ratio_default,
        _upload_reference_images_for_node,
        _tensors_to_uploaded_urls,
        _pil_to_tensor,
        _cat_image_batches_safe,
        _run_three_view_jobs,
        _THREE_VIEW_SCOPE_OPTIONS,
        _runtime_results_payload,
        _clear_runtime_results,
        _error_img,
        _return_images_with_ui_preview,
        _HAS_PROMPT_SERVER,
        PromptServer,
        aiohttp_web,
    )
except Exception as e:
    raise RuntimeError(f"banana_triple_view_node.py 依赖 banana_node.py，请确认 banana_node.py 已正确安装: {e}") from e


PLUGIN_VERSION = "7.12.0-comfyui-only-veo-direct"

EDITOR_ROUTE = "/banana/triple-view-editor"
MANIFEST_ROUTE = "/banana/ecommerce-manifest"
CONFIG_ROUTE = "/banana/ecommerce-prompt-config"
CONFIG_DEFAULTS_ROUTE = "/banana/ecommerce-prompt-config/defaults"
RUNTIME_ROUTE = "/banana/ecommerce-runtime-results"
RUNTIME_CLEAR_ROUTE = "/banana/ecommerce-runtime-clear"
AUTOMATION_SELECT_FOLDER_ROUTE = "/banana/automation/select-folder"
AUTOMATION_PREVIEW_ROUTE = "/banana/automation/preview"
AUTOMATION_HISTORY_ROUTE = "/banana/automation/history"
AUTOMATION_HISTORY_CLEAR_ROUTE = "/banana/automation/history-clear"
AUTOMATION_HISTORY_FILE = "banana_automation_history.json"
_AUTOMATION_HISTORY_LOCK = threading.Lock()
_AUTOMATION_HISTORY_MAX_ITEMS = 500

MODE_OPTIONS: Dict[str, str] = {
    "服装迁移": "fashion_replace",
    "服装穿戴": "model_tryon",
    "动作迁移": "pose_transfer",
    "印花面料迁移": "texture_pattern_transfer",
    "单品三视图": "product_three_view",
    "白底主图精修": "white_bg_refine",
}

DELETED_MODE_KEYS = {
    "accessory_tryon",
    "model_create",
    "ai_design",
    "ai_designer",
    "designer_theme",
}

FIELD_KEYS = [
    "image_roles",
    "global_prompt",
    "front_prompt",
    "side_prompt",
    "back_prompt",
    "consistency_prompt",
    "negative_prompt",
]

MODE_EXTRA_KEYS = [
    "preview_urls",
    "previewUrls",
]

TOP_LEVEL_KEEP_KEYS = [
    "plugin_version",
    "version",
    "description",
    "preview_base_url",
    "preview_ext",
    "background_url",
    "mode_meta",
    "preview_urls",
    "previewUrls",
]

ECOMMERCE_DEFAULTS: Dict[str, Any] = {
    "display_name": "🍌 Banana｜图像生成",
    "category": "AI电商/提示词模板",
    "output_node": True,
    "default_prompt_template": "动作迁移",
    "default_mode": "动作迁移",
    "default_model": "banano2",
    "default_image_size": "4K",
    "default_aspect_ratio": "16:9 (横屏宽幅)",
    "prompt_store_path": "banana_ecommerce_prompts.json",
    "editor_route": EDITOR_ROUTE,
    "editor_html": "web/banana_prompt_editor.html",
    "theme": "冬之韵",
    "theme_en": "Winter Rhyme",
    "theme_accent": "#8fc7ff",
    "theme_deep": "#315d8f",
    "theme_bg": "linear-gradient(135deg, #eef7ff 0%, #f8fbff 48%, #fff7fb 100%)",
    "preview_base_url": "https://img.hrio.site/plugins2/previews",
    "preview_ext": "png",
    "mode_options": MODE_OPTIONS,
    "optional_image_slots": 10,
}

DEFAULT_MODES: Dict[str, Dict[str, Any]] = {
    "fashion_replace": {
        "image_roles": "Image A（reference_image_1）= 服装/穿搭来源图；Image B（reference_image_2）= 目标样板/模特/人台。A 只提供服装本身；B 只提供被换装的主体。",
        "global_prompt": "模式：服装迁移。请保留 Image B 的主体身份、脸部、发型、身体比例、肤色、五官与主体结构不变，仅将 Image A 中的服装、颜色、面料、图案、辅料、结构与配件完整替换到 Image B 身上。如果 B 原本已有穿戴，则全部替换。三次请求使用完全相同的参考图输入，只改变视角提示词。",
        "front_prompt": "请输出【正面图】。主体正面朝向镜头，完整展示换装后的服装正面信息，清晰展示领口、胸前、门襟、袖型、下摆、正面印花、图案、文字等。",
        "side_prompt": "请输出【侧面图】。主体严格 90 度侧身，完整展示服装侧面轮廓、厚度、垂坠、侧缝、袖型侧面、层次关系。",
        "back_prompt": "请输出【背面图】。主体背对镜头，完整展示服装背面结构、后领、后片、背部图案、文字、后摆等细节。",
        "consistency_prompt": "三张图必须是同一个目标样板、同一套服装、同一灯光、同一白底、同一镜头距离。只允许视角变化，不允许主体身份、服装款式、颜色、面料、比例发生变化。",
        "negative_prompt": "禁止保留目标样板原有服装；禁止引入额外人物、背景道具、复杂场景；禁止拼图、多视图排版、标注、水印、logo、多人、错误肢体。",
        "preview_urls": {},
    },
    "model_tryon": {
        "image_roles": "Image A（reference_image_1）= 目标模特/人台/人物，是唯一主体来源；Image B 到最后一张图（reference_image_2...reference_image_N）= 全部需要穿戴到 A 身上的商品，可以是服装、鞋子、包袋、帽子、首饰、眼镜、配饰或其他穿戴物。",
        "global_prompt": "模式：服装穿戴。请严格保留 Image A 中模特/人物的身份、五官、脸型、发型、肤色、身体比例、年龄感、气质与姿态基础不变。请将 Image B 到最后一张图中的所有商品全部自然穿戴到 Image A 模特身上。除图片1外，其余上传图片全部都作为穿戴商品来源，不得遗漏。若 A 模特身上已有同类服装、鞋包或配饰，必须用后续参考图商品替换原有穿戴；若 A 模特身上没有对应品类，则自然添加到正确身体部位。",
        "front_prompt": "请输出【正面图】。A 模特正面朝向镜头，完整展示所有穿戴商品的正面效果，包括上装、下装、鞋、包、帽子、首饰、眼镜等可见品类。",
        "side_prompt": "请输出【侧面图】。A 模特严格 90 度侧身，展示所有穿戴商品的侧面轮廓、厚度、垂坠、层次、鞋包配饰位置和真实遮挡关系。",
        "back_prompt": "请输出【背面图】。A 模特背对镜头，展示所有穿戴商品的背面结构、后片、背部图案、包袋背面、鞋跟或配饰背面细节。",
        "consistency_prompt": "三张图必须是同一个 A 模特、同一组 B-N 商品、同一套穿戴关系、同一灯光、同一白底、同一镜头距离。只允许视角变化，不允许模特身份、商品款式、颜色、材质、Logo、文字、图案、比例发生变化。",
        "negative_prompt": "禁止遗漏后续参考图商品；禁止把商品图里的背景、道具、其他人物带入；禁止新增参考图不存在的服装、配饰、文字或 Logo；禁止多人物、拼图、三联图、九宫格、水印、低清晰度、身体畸形、穿帮、错位、漂浮。",
        "preview_urls": {},
    },
    "pose_transfer": {
        "image_roles": "Image A（reference_image_1）= 主体内容唯一来源；Image B（reference_image_2）= 动作、姿态、肢体语言与表现手法参考。",
        "global_prompt": "模式：动作迁移。请严格保留 Image A 中主体的所有外观特征，包括造型、颜色、比例、五官、服饰、文字、Logo、图案、材质、结构与细节完全不变，仅让主体做出 Image B 中展示的动作、姿态和肢体语言。生成图片的所有视觉内容必须且只能来自 A；B 只提供动作参考，不得引入 B 的任何具体内容元素。",
        "front_prompt": "请输出【正面图】。在正面视角下重建动作迁移效果，保持主体仍然完全来自 A，动作来自 B。",
        "side_prompt": "请输出【侧面图】。在严格侧面视角下重建同一动作与肢体语言，展示清晰侧面动态轮廓。",
        "back_prompt": "请输出【背面图】。在背面视角下重建同一动作与肢体语言，展示背面动作结构，主体外观仍严格来自 A。",
        "consistency_prompt": "三张图必须保持同一个 A 图主体与同一个动作迁移逻辑，只改变观看视角。",
        "negative_prompt": "严禁引入 B 图中的具体文字、图案、角色、装饰、配色方案或额外元素；严禁主体设计漂移、服装变化、五官变化、比例变化。",
        "preview_urls": {},
    },
    "texture_pattern_transfer": {
        "image_roles": "Image A（reference_image_1）= 印花、面料、花型来源图；Image B（reference_image_2）= 目标服装版型或目标商品。",
        "global_prompt": "模式：印花面料迁移。请保留 Image B 的服装版型、商品结构、主体轮廓与样板基础不变，仅将 Image A 中的印花、面料肌理、图案风格与色彩方案迁移到 Image B 对应表面。",
        "front_prompt": "请输出【正面图】。正面清晰展示迁移后的面料、印花覆盖效果与版型关系。",
        "side_prompt": "请输出【侧面图】。侧面清晰展示面料垂坠、厚度与印花延展逻辑。",
        "back_prompt": "请输出【背面图】。背面清晰展示印花、面料在背部区域的完整延续效果。",
        "consistency_prompt": "三张图必须保持同一个目标版型与同一套迁移后的面料、印花方案，只改变视角。",
        "negative_prompt": "禁止改动目标版型轮廓；禁止引入额外人物、场景、道具、拼贴、标注、水印、logo。",
        "preview_urls": {},
    },
    "product_three_view": {
        "image_roles": "Image A（reference_image_1）= 商品唯一内容来源。",
        "global_prompt": "模式：单品三视图。请以 Image A 的商品为唯一内容来源，生成标准电商三视图。必须严格保留商品的结构、比例、颜色、材质、五金、图案、文字与所有识别性细节。",
        "front_prompt": "请输出【正面图】。商品居中，正面朝向镜头，完整展示主视觉面。",
        "side_prompt": "请输出【侧面图】。商品严格侧向展示，清晰显示厚度、侧面结构与轮廓。",
        "back_prompt": "请输出【背面图】。商品背对镜头，清晰展示背部结构、接口、缝线、后部细节。",
        "consistency_prompt": "三张图必须是同一个商品，仅改变角度，不改变任何造型、材质、颜色、印花、文字与比例。",
        "negative_prompt": "禁止支架、场景化背景、道具、额外装饰、拼版、多商品、文字标识、水印、logo。",
        "preview_urls": {},
    },
    "white_bg_refine": {
        "image_roles": "Image A（reference_image_1）= 主体来源图。",
        "global_prompt": "模式：白底主图精修。请以 Image A 为主体来源，在不改变核心商品或主体设计的前提下，输出更适合电商发布的正面、侧面、背面白底图。重点提升清晰度、材质质感、边缘干净度与商业观感。",
        "front_prompt": "请输出【正面图】。主体正面清晰、白底干净、商业质感强。",
        "side_prompt": "请输出【侧面图】。主体侧面结构清楚，边缘锐利，便于详情页使用。",
        "back_prompt": "请输出【背面图】。主体背面信息完整，光影干净，适合白底电商展示。",
        "consistency_prompt": "三张图保持同一主体与同一白底棚拍风格，只改变视角。",
        "negative_prompt": "禁止场景化背景、道具、夸张特效、拼版、文字标识、水印、logo。",
        "preview_urls": {},
    },
}


def _strip_deleted_modes(options: Dict[str, str]) -> Dict[str, str]:
    return {
        str(k): str(v)
        for k, v in (options or {}).items()
        if str(v) not in DELETED_MODE_KEYS
    }


def _ecommerce_manifest() -> Dict[str, Any]:
    raw = _MANIFEST.get("ecommerce_three_view", {}) or {}
    merged = copy.deepcopy(ECOMMERCE_DEFAULTS)

    if isinstance(raw, dict):
        for k, v in raw.items():
            if k == "mode_options" and isinstance(v, dict) and v:
                merged[k] = _strip_deleted_modes(v)
            else:
                merged[k] = v

    merged["display_name"] = "🍌 Banana｜图像生成"
    merged["category"] = "AI电商/提示词模板"
    merged["editor_route"] = EDITOR_ROUTE
    merged["theme"] = "冬之韵"
    merged["theme_en"] = "Winter Rhyme"
    merged["theme_accent"] = "#8fc7ff"
    merged["theme_deep"] = "#315d8f"
    merged["theme_bg"] = "linear-gradient(135deg, #eef7ff 0%, #f8fbff 48%, #fff7fb 100%)"

    merged["mode_options"] = _strip_deleted_modes(merged.get("mode_options") or MODE_OPTIONS)
    if not merged["mode_options"]:
        merged["mode_options"] = copy.deepcopy(MODE_OPTIONS)

    return merged


def _prompt_config_path() -> str:
    cfg = _ecommerce_manifest()
    filename = str(cfg.get("prompt_store_path") or "banana_ecommerce_prompts.json").strip()
    filename = filename.replace("\\", "/").split("/")[-1] or "banana_ecommerce_prompts.json"
    return os.path.join(MODULE_DIR, filename)


def _manifest_mode_options() -> Dict[str, str]:
    return _strip_deleted_modes(_ecommerce_manifest().get("mode_options") or MODE_OPTIONS)


def _mode_actual_from_display(value: Any, hidden_key: Any = "") -> str:
    hidden = str(hidden_key or "").strip()
    options = _manifest_mode_options()

    if hidden and hidden in set(options.values()) and hidden not in DELETED_MODE_KEYS:
        return hidden

    raw = str(value or "").strip()

    if raw in options:
        return options[raw]

    for _, actual in options.items():
        if raw == actual:
            return actual

    default_display = str(
        _ecommerce_manifest().get("default_prompt_template")
        or _ecommerce_manifest().get("default_mode")
        or "动作迁移"
    )
    return options.get(default_display, "pose_transfer")


def _mode_display_from_actual(actual_value: Any) -> str:
    actual = str(actual_value or "").strip()

    for display, value in _manifest_mode_options().items():
        if str(value).strip() == actual:
            return display

    return str(
        _ecommerce_manifest().get("default_prompt_template")
        or _ecommerce_manifest().get("default_mode")
        or "动作迁移"
    )


def _field_dict_from_any(raw: Any, fallback_key: str) -> Dict[str, Any]:
    base = copy.deepcopy(DEFAULT_MODES.get(fallback_key, DEFAULT_MODES["pose_transfer"]))

    if isinstance(raw, dict):
        for key in FIELD_KEYS:
            if isinstance(raw.get(key), str):
                base[key] = raw[key]

        for key in MODE_EXTRA_KEYS:
            value = raw.get(key)
            if isinstance(value, dict):
                base[key] = copy.deepcopy(value)

        if "previewUrls" in base and "preview_urls" not in base:
            base["preview_urls"] = copy.deepcopy(base.get("previewUrls") or {})

    elif isinstance(raw, str) and raw.strip():
        base["global_prompt"] = raw.strip()

    for key in FIELD_KEYS:
        base.setdefault(key, "")

    if not isinstance(base.get("preview_urls"), dict):
        base["preview_urls"] = {}

    return base


def _default_prompt_config() -> Dict[str, Any]:
    cfg = _ecommerce_manifest()
    modes: Dict[str, Dict[str, Any]] = {}

    for actual in set(list(_manifest_mode_options().values()) + list(DEFAULT_MODES.keys())):
        if actual in DELETED_MODE_KEYS:
            continue
        if actual in DEFAULT_MODES:
            item = copy.deepcopy(DEFAULT_MODES[actual])
            item.setdefault("preview_urls", {})
            modes[actual] = item

    return {
        "plugin_version": PLUGIN_VERSION,
        "version": 10,
        "description": "Banana 图像生成提示词模板配置。mode/prompt_template 表示提示词模板；model 表示大模型，两者互不干扰。主题：冬之韵。",
        "preview_base_url": cfg.get("preview_base_url", "https://img.hrio.site/plugins2/previews"),
        "preview_ext": cfg.get("preview_ext", "png"),
        "background_url": cfg.get("background_url", ""),
        "mode_options": _manifest_mode_options(),
        "mode_meta": {},
        "preview_urls": {},
        "modes": modes,
        "prompts": {},
    }


def _read_prompt_config() -> Dict[str, Any]:
    path = _prompt_config_path()
    default_cfg = _default_prompt_config()

    if not os.path.exists(path):
        return default_cfg

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        logger.warning(f"读取 {os.path.basename(path)} 失败，使用默认配置: {e}")
        return default_cfg

    if not isinstance(raw, dict):
        return default_cfg

    merged = copy.deepcopy(default_cfg)

    for key in TOP_LEVEL_KEEP_KEYS:
        if key in raw:
            merged[key] = copy.deepcopy(raw[key])

    raw_options = raw.get("mode_options")
    if isinstance(raw_options, dict) and raw_options:
        merged["mode_options"] = _strip_deleted_modes(raw_options)

    raw_modes = raw.get("modes")
    if isinstance(raw_modes, dict):
        for actual, fields in raw_modes.items():
            actual = str(actual)
            if actual in DELETED_MODE_KEYS:
                continue
            merged["modes"][actual] = _field_dict_from_any(fields, actual)

    old_prompts = raw.get("prompts")
    if isinstance(old_prompts, dict):
        merged["prompts"] = old_prompts

    top_preview_urls = raw.get("preview_urls") or raw.get("previewUrls")
    if isinstance(top_preview_urls, dict):
        merged["preview_urls"] = copy.deepcopy(top_preview_urls)

        for mode_key, urls in top_preview_urls.items():
            if not isinstance(urls, dict):
                continue

            mode_key = str(mode_key)
            if mode_key in DELETED_MODE_KEYS:
                continue

            if mode_key not in merged["modes"]:
                merged["modes"][mode_key] = _field_dict_from_any({}, mode_key)

            if not isinstance(merged["modes"][mode_key].get("preview_urls"), dict):
                merged["modes"][mode_key]["preview_urls"] = {}

            for view in ["front", "side", "back"]:
                if urls.get(view):
                    merged["modes"][mode_key]["preview_urls"][view] = urls.get(view)

    return merged


def _save_prompt_config(data: Any) -> Dict[str, Any]:
    current = _read_prompt_config()

    if not isinstance(data, dict):
        raise RuntimeError("保存失败：请求体必须是 JSON 对象")

    saved = copy.deepcopy(current)

    for key in [
        "description",
        "preview_base_url",
        "preview_ext",
        "background_url",
        "mode_meta",
        "preview_urls",
        "previewUrls",
    ]:
        if key in data:
            save_key = "preview_urls" if key == "previewUrls" else key
            saved[save_key] = copy.deepcopy(data[key])

    if isinstance(data.get("mode_options"), dict):
        saved["mode_options"] = _strip_deleted_modes(data["mode_options"])

    if isinstance(data.get("modes"), dict):
        for actual, fields in data["modes"].items():
            actual = str(actual)
            if actual in DELETED_MODE_KEYS:
                continue

            saved.setdefault("modes", {})[actual] = _field_dict_from_any(fields, actual)

    if isinstance(saved.get("preview_urls"), dict):
        for mode_key, urls in saved["preview_urls"].items():
            if not isinstance(urls, dict):
                continue

            mode_key = str(mode_key)
            if mode_key in DELETED_MODE_KEYS:
                continue

            saved.setdefault("modes", {})
            if mode_key not in saved["modes"]:
                saved["modes"][mode_key] = _field_dict_from_any({}, mode_key)

            saved["modes"][mode_key].setdefault("preview_urls", {})

            for view in ["front", "side", "back"]:
                if urls.get(view):
                    saved["modes"][mode_key]["preview_urls"][view] = urls.get(view)

    preview_urls = {}
    for mode_key, fields in (saved.get("modes") or {}).items():
        if not isinstance(fields, dict):
            continue

        urls = fields.get("preview_urls")
        if isinstance(urls, dict):
            preview_urls[mode_key] = {
                "front": urls.get("front", ""),
                "side": urls.get("side", ""),
                "back": urls.get("back", ""),
            }

    saved["preview_urls"] = preview_urls

    saved["plugin_version"] = PLUGIN_VERSION
    saved["version"] = int(saved.get("version") or 0) + 1

    path = _prompt_config_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(saved, f, ensure_ascii=False, indent=2)

    logger.success(f"Banana 图像生成提示词模板配置已保存: {path}")
    return saved


def _mode_fields(mode_actual: str) -> Dict[str, Any]:
    cfg = _read_prompt_config()
    modes = cfg.get("modes") or {}
    fields = modes.get(mode_actual)

    if not isinstance(fields, dict):
        fields = DEFAULT_MODES.get(mode_actual, DEFAULT_MODES["pose_transfer"])

    return _field_dict_from_any(fields, mode_actual)


def _compose_prompt(mode_actual: str, view_key: str) -> str:
    fields = _mode_fields(mode_actual)

    view_field = {
        "front": "front_prompt",
        "side": "side_prompt",
        "back": "back_prompt",
    }.get(view_key, "front_prompt")

    parts = [
        f"参考图角色：{fields.get('image_roles', '')}",
        f"全局任务：{fields.get('global_prompt', '')}",
        f"视角任务：{fields.get(view_field, '')}",
        f"一致性要求：{fields.get('consistency_prompt', '')}",
        f"负面约束：{fields.get('negative_prompt', '')}",
        "输出要求：只输出单张图片，不要拼图，不要三联图，不要九宫格，不要文字标注，不要水印。主体边缘清晰，光影干净，结构准确。",
    ]

    return "\n\n".join([p for p in parts if str(p or "").strip()])



# -----------------------------------------------------------------------------
# 自动化：后端扫描输入根目录 -> 按子文件夹数字序号横向聚合 -> 并发执行不同序号组。
# -----------------------------------------------------------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _safe_int_local(value: Any, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        out = int(float(str(value).strip()))
    except Exception:
        out = int(default)
    if min_value is not None:
        out = max(int(min_value), out)
    if max_value is not None:
        out = min(int(max_value), out)
    return out


def _safe_bool_local(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _extract_sequence(folder_name: str) -> str:
    parts = re.findall(r"\d+", str(folder_name or ""))
    return "".join(parts) if parts else ""


def _clean_path_list(values: Any, max_count: int = 10) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen = set()
    for item in values:
        path = str(item or "").strip().strip('"')
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
        if len(out) >= max_count:
            break
    return out


def _clean_sequence_list(values: Any, max_count: int = 9999) -> List[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    out: List[str] = []
    seen = set()
    for item in values:
        raw = str(item or "").strip().strip('"')
        if not raw:
            continue
        seq = _extract_sequence(raw) or raw
        if not seq or seq in seen:
            continue
        seen.add(seq)
        out.append(seq)
        if len(out) >= max_count:
            break
    return out


def _scan_input_root(root: str) -> List[Dict[str, Any]]:
    """
    只支持「根目录直放图片模式」。

    输入示例：
        input_root_01/001.png
        input_root_01/002.png
        input_root_02/001.png
        input_root_02/002.png

    扫描规则：
    - 只扫描 input_root 下的直接图片文件；
    - 不扫描 001_截图/ 这类子文件夹；
    - 从图片文件名中贪婪提取所有数字并拼接作为序号；
    - 相同序号会在多个 input_root 之间横向聚合。
    """
    items: List[Dict[str, Any]] = []
    root = str(root or "").strip()
    if not root or not os.path.isdir(root):
        return items

    try:
        names = sorted(os.listdir(root))
    except Exception:
        return items

    for name in names:
        full = os.path.join(root, name)
        if not os.path.isfile(full):
            continue

        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXTS:
            continue

        stem = os.path.splitext(name)[0]
        seq = _extract_sequence(stem)
        if not seq:
            continue

        items.append({
            "source_type": "root_image",
            "file_name": name,
            "image_path": full,
            "sequence": seq,
        })

    return items


def _sequence_sort_key(seq: str):
    text = str(seq or "")
    try:
        return (0, int(text), len(text), text)
    except Exception:
        return (1, 0, len(text), text)


def _build_sequence_groups(input_roots: List[str], output_root: str = "", require_all_roots_present: bool = False) -> List[Dict[str, Any]]:
    """
    只按根目录图片文件名分组。

    每个 input_root 是一个输入槽位；执行时每个序号组最多从 10 个 input_root 中各取一张同序号图片，
    例如 001 组会收集：
        input_root_01/001.png
        input_root_02/001.png
        ...
        input_root_10/001.png
    """
    group_map: Dict[str, List[Dict[str, Any]]] = {}
    root_count = len(input_roots)

    for root_index, root in enumerate(input_roots):
        for item in _scan_input_root(root):
            seq = item["sequence"]
            group_map.setdefault(seq, []).append({
                "root_index": root_index,
                "root_path": root,
                "source_type": "root_image",
                "file_name": item["file_name"],
                "image_path": item["image_path"],
                "sequence": seq,
            })

    groups: List[Dict[str, Any]] = []
    for seq in sorted(group_map.keys(), key=_sequence_sort_key):
        items = sorted(group_map[seq], key=lambda x: int(x.get("root_index") or 0))
        present_roots = {int(x.get("root_index") or 0) for x in items}
        if require_all_roots_present and len(present_roots) < root_count:
            continue
        run_dir = os.path.join(str(output_root or ""), f"output_{seq}", "run_01") if output_root else ""
        groups.append({
            "sequence": seq,
            "items": items,
            "output_dir": run_dir,
            "present_root_count": len(present_roots),
            "expected_root_count": root_count,
        })
    return groups


def _collect_images_for_group(items: List[Dict[str, Any]], max_count: int = 10) -> List[str]:
    """收集某个序号组里的直接图片路径，按 input_root 顺序排列。"""
    paths: List[str] = []
    for item in sorted(items or [], key=lambda x: int(x.get("root_index") or 0)):
        image_path = str(item.get("image_path") or "")
        if not image_path or not os.path.isfile(image_path):
            continue
        ext = os.path.splitext(image_path)[1].lower()
        if ext in IMAGE_EXTS:
            paths.append(image_path)
            if len(paths) >= max_count:
                return paths
    return paths[:max_count]

def _automation_payload_from_string(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _automation_enabled(raw: Any) -> bool:
    data = _automation_payload_from_string(raw)
    return bool(data) and _safe_bool_local(data.get("enabled"), False)


def _normalize_automation_payload(raw: Any) -> Dict[str, Any]:
    data = _automation_payload_from_string(raw)
    input_roots = _clean_path_list(data.get("input_roots") or data.get("inputFolders") or data.get("input_folders"), 10)
    output_root = str(data.get("output_root") or data.get("outputRoot") or "").strip()
    group_concurrency = _safe_int_local(data.get("group_concurrency", data.get("groupConcurrency", 3)), 3, 1, 10)
    max_images_per_group = _safe_int_local(data.get("max_images_per_group", data.get("maxImagesPerGroup", 10)), 10, 1, 10)
    require_all = _safe_bool_local(data.get("require_all_roots_present"), False)
    run_sequences = _clean_sequence_list(
        data.get("run_sequences")
        or data.get("target_sequences")
        or data.get("sequences")
        or data.get("run_sequence")
        or data.get("runSequence")
        or data.get("selected_sequence")
        or data.get("sequence")
    )
    run_view = str(data.get("run_view") or data.get("view") or "").strip()
    run_mode = str(data.get("run_mode") or data.get("action") or "").strip()

    return {
        "enabled": _safe_bool_local(data.get("enabled"), False),
        "version": str(data.get("version") or "7.10.0"),
        "input_roots": input_roots,
        "output_root": output_root,
        "group_concurrency": group_concurrency,
        "max_input_roots": 10,
        "max_images_per_group": max_images_per_group,
        "extract_rule": "greedy_digits_join_all",
        "collect_images_mode": "root_images_group_by_filename_sequence",
        "collect_mode": "root_images_group_by_filename_sequence",
        "require_all_roots_present": require_all,
        "save_images": _safe_bool_local(data.get("save_images"), True),
        "save_video": _safe_bool_local(data.get("save_video"), False),
        "run_sequences": run_sequences,
        "run_view": run_view,
        "run_mode": run_mode,
        "video_filename": str(data.get("video_filename") or "result.mp4"),
        "image_filenames": data.get("image_filenames") if isinstance(data.get("image_filenames"), dict) else {
            "front": "front.png",
            "side": "side.png",
            "back": "back.png",
        },
    }


def _automation_preview(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _normalize_automation_payload(payload)
    input_roots = cfg["input_roots"]
    output_root = cfg["output_root"]
    groups = _build_sequence_groups(
        input_roots,
        output_root=output_root,
        require_all_roots_present=bool(cfg.get("require_all_roots_present")),
    )

    return {
        "ok": True,
        "input_roots": input_roots,
        "output_root": output_root,
        "group_count": len(groups),
        "groups": groups,
    }


def _select_folder_with_tkinter() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askdirectory(title="选择 Banana 自动化文件夹")
    finally:
        root.destroy()
    return str(path or "").strip()


def _load_image_tensors_from_paths(paths: List[str]) -> List[Any]:
    tensors: List[Any] = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        tensors.append(_pil_to_tensor(img))
    return tensors


def _save_tensor_image(tensor: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = tensor.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    arr = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _write_text_file(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(text or ""))


def _automation_history_path() -> str:
    return os.path.join(MODULE_DIR, AUTOMATION_HISTORY_FILE)


def _utc_now_ms() -> int:
    return int(time.time() * 1000)


def _read_automation_history() -> Dict[str, Any]:
    path = _automation_history_path()
    if not os.path.exists(path):
        return {
            "ok": True,
            "version": PLUGIN_VERSION,
            "updated_at_ms": 0,
            "count": 0,
            "items": [],
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {
            "ok": True,
            "version": PLUGIN_VERSION,
            "updated_at_ms": 0,
            "count": 0,
            "items": [],
        }

    if not isinstance(data, dict):
        data = {}
    items = data.get("items") if isinstance(data.get("items"), list) else []
    data["ok"] = True
    data["version"] = str(data.get("version") or PLUGIN_VERSION)
    data["updated_at_ms"] = int(data.get("updated_at_ms") or 0)
    data["count"] = len(items)
    data["items"] = items[-_AUTOMATION_HISTORY_MAX_ITEMS:]
    return data


def _clear_automation_history() -> Dict[str, Any]:
    payload = {
        "ok": True,
        "version": PLUGIN_VERSION,
        "updated_at_ms": _utc_now_ms(),
        "count": 0,
        "items": [],
    }
    with _AUTOMATION_HISTORY_LOCK:
        with open(_automation_history_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def _history_existing_files(output_dir: str) -> Dict[str, str]:
    output_dir = str(output_dir or "")
    names = ["front.png", "side.png", "back.png", "result.mp4", "run_info.json", "error.txt"]
    out: Dict[str, str] = {}
    for name in names:
        path = os.path.join(output_dir, name) if output_dir else ""
        if path and os.path.exists(path):
            out[name] = path
    return out


def _append_automation_history_record(record: Dict[str, Any]) -> None:
    if not isinstance(record, dict):
        return

    item = copy.deepcopy(record)
    for key in ("front", "side", "back", "batch", "tensor", "image"):
        item.pop(key, None)

    output_dir = str(item.get("output_dir") or "")
    item.setdefault("output_files", _history_existing_files(output_dir))
    item.setdefault("created_at_ms", _utc_now_ms())
    item.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    item.setdefault("plugin_version", PLUGIN_VERSION)

    with _AUTOMATION_HISTORY_LOCK:
        data = _read_automation_history()
        items = data.get("items") if isinstance(data.get("items"), list) else []
        items.append(item)
        items = items[-_AUTOMATION_HISTORY_MAX_ITEMS:]
        payload = {
            "ok": True,
            "version": PLUGIN_VERSION,
            "updated_at_ms": _utc_now_ms(),
            "count": len(items),
            "items": items,
        }
        with open(_automation_history_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def _run_automation_one_group(
    *,
    group: Dict[str, Any],
    cfg: Dict[str, Any],
    api_key: str,
    model: str,
    image_size: str,
    aspect_ratio: str,
    prompts: Dict[str, str],
    labels: str,
    cache_key: str,
    auto_retry_until_success: bool,
    max_retry_per_view: int,
    retry_interval_sec: float,
    generate_scope: str = "全部并发生成",
) -> Dict[str, Any]:
    seq = str(group.get("sequence") or "")
    run_dir = str(group.get("output_dir") or os.path.join(cfg["output_root"], f"output_{seq}", "run_01"))
    os.makedirs(run_dir, exist_ok=True)

    try:
        image_paths = _collect_images_for_group(group.get("items") or [], int(cfg.get("max_images_per_group") or 10))
        if not image_paths:
            raise RuntimeError(f"序号 {seq} 没有找到可用图片")

        tensors = _load_image_tensors_from_paths(image_paths)
        upload_dir = _cfg_or_manifest("upload_dir", "uploads/images")
        image_urls = _tensors_to_uploaded_urls(tensors, api_key, upload_dir)

        result = _run_three_view_jobs(
            api_key=api_key,
            model=model,
            image_size=image_size,
            aspect_ratio=aspect_ratio,
            image_urls=image_urls,
            prompts=prompts,
            labels_prefix=f"{labels}-自动化{seq}-" if labels else f"自动化{seq}-",
            generate_scope=generate_scope,
            cache_key=f"{cache_key}:{seq}",
            auto_retry_until_success=auto_retry_until_success,
            max_retry_per_view=max_retry_per_view,
            retry_interval_sec=retry_interval_sec,
        )

        image_names = cfg.get("image_filenames") or {}
        if bool(cfg.get("save_images", True)):
            _save_tensor_image(result["front"], os.path.join(run_dir, str(image_names.get("front") or "front.png")))
            _save_tensor_image(result["side"], os.path.join(run_dir, str(image_names.get("side") or "side.png")))
            _save_tensor_image(result["back"], os.path.join(run_dir, str(image_names.get("back") or "back.png")))

        meta = {
            "sequence": seq,
            "ok": True,
            "node_type": "image_template",
            "output_dir": run_dir,
            "input_image_count": len(image_paths),
            "uploaded_image_count": len(image_urls),
            "source_images": image_paths,
            "errors_by_key": result.get("errors_by_key") or {},
            "generate_scope": result.get("generate_scope") or generate_scope,
            "model": model,
            "image_size": image_size,
            "aspect_ratio": aspect_ratio,
            "labels": labels,
        }
        _write_text_file(os.path.join(run_dir, "run_info.json"), json.dumps(meta, ensure_ascii=False, indent=2))
        _append_automation_history_record(meta)
        return {
            **meta,
            "front": result.get("front"),
            "side": result.get("side"),
            "back": result.get("back"),
            "batch": result.get("batch"),
        }

    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        _write_text_file(os.path.join(run_dir, "error.txt"), err)
        logger.error(f"自动化序号 {seq} 失败: {e}")
        fail_meta = {
            "sequence": seq,
            "ok": False,
            "node_type": "image_template",
            "output_dir": run_dir,
            "error": str(e),
            "model": model,
            "image_size": image_size,
            "aspect_ratio": aspect_ratio,
            "labels": labels,
            "generate_scope": generate_scope,
        }
        _append_automation_history_record(fail_meta)
        return fail_meta

def _register_ecommerce_routes() -> None:
    if not _HAS_PROMPT_SERVER or PromptServer is None or aiohttp_web is None:
        return

    routes = PromptServer.instance.routes

    def _json_response(payload: Any, status: int = 200):
        return aiohttp_web.json_response(
            payload,
            status=status,
            dumps=lambda x: json.dumps(x, ensure_ascii=False),
        )

    def _route_exists(method: str, path: str) -> bool:
        method = str(method or "GET").upper()
        path = str(path or "").strip()
        try:
            resources = list(routes.resources())
        except Exception:
            resources = []

        for resource in resources:
            canonical = str(getattr(resource, "canonical", "") or "")
            if canonical != path:
                continue
            try:
                resource_routes = list(resource)
            except Exception:
                resource_routes = []
            for route in resource_routes:
                route_method = str(getattr(route, "method", "") or "").upper()
                if route_method == method or route_method == "*":
                    return True
                # aiohttp 给 GET 自动注册 HEAD；HEAD 存在时也说明这个 path 已经占用过。
                if method == "GET" and route_method in {"GET", "HEAD"}:
                    return True
        return False

    def _safe_add_route(method: str, path: str, handler) -> None:
        method = str(method or "GET").upper()
        path = str(path or "").strip()
        if not path:
            return
        if _route_exists(method, path):
            logger.info(f"Banana 路由已存在，跳过重复注册: {method} {path}")
            return
        try:
            decorator = getattr(routes, method.lower())
            decorator(path)(handler)
            logger.success(f"Banana 路由已注册: {method} {path}")
        except Exception as e:
            msg = str(e)
            if "already registered" in msg or "will never be executed" in msg or "Duplicate" in msg:
                logger.info(f"Banana 路由重复，已跳过: {method} {path}")
                return
            logger.warning(f"Banana 路由注册失败: {method} {path} | {e}")

    async def _banana_ecommerce_manifest_get(request):
        payload = {
            "ok": True,
            "plugin_version": PLUGIN_VERSION,
            "editor_route": EDITOR_ROUTE,
            "editor_alias_routes": ["/banana/prompt-editor", "/banana/ecommerce-editor", "/ai-ecommerce/editor"],
            "ecommerce_three_view": _ecommerce_manifest(),
            "mode_options": _manifest_mode_options(),
            "prompt_config_path": os.path.basename(_prompt_config_path()),
        }
        return _json_response(payload)

    async def _banana_ecommerce_prompt_config_get(request):
        return _json_response(_read_prompt_config())

    async def _banana_ecommerce_prompt_config_post(request):
        try:
            data = await request.json()
            saved = _save_prompt_config(data)
            return _json_response(
                {
                    "ok": True,
                    "config": saved,
                    "prompt_config": saved,
                }
            )
        except Exception as e:
            return _json_response(
                {
                    "ok": False,
                    "error": str(e),
                },
                status=500,
            )

    async def _banana_ecommerce_prompt_config_defaults(request):
        return _json_response(_default_prompt_config())

    async def _banana_triple_view_editor(request):
        cfg = _ecommerce_manifest()
        rel = str(cfg.get("editor_html") or "web/banana_prompt_editor.html").strip()
        rel = rel.replace("\\", "/").lstrip("/")
        html_path = os.path.join(MODULE_DIR, rel)

        if not os.path.exists(html_path):
            return aiohttp_web.Response(
                status=404,
                text=f"找不到前端面板文件: {html_path}",
                content_type="text/plain",
                headers={"Cache-Control": "no-store"},
            )

        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        return aiohttp_web.Response(
            text=html,
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    async def _banana_ecommerce_runtime_results_get(request):
        payload = _runtime_results_payload()
        if isinstance(payload, dict):
            payload["editor_route"] = EDITOR_ROUTE
        return _json_response(payload)

    async def _banana_ecommerce_runtime_clear_post(request):
        return _json_response(_clear_runtime_results())

    async def _banana_automation_select_folder_post(request):
        try:
            path = _select_folder_with_tkinter()
            return _json_response({"ok": bool(path), "path": path})
        except Exception as e:
            return _json_response({"ok": False, "error": str(e)}, status=500)

    async def _banana_automation_preview_post(request):
        try:
            data = await request.json()
            return _json_response(_automation_preview(data))
        except Exception as e:
            return _json_response({"ok": False, "error": str(e)}, status=500)

    async def _banana_automation_history_get(request):
        try:
            return _json_response(_read_automation_history())
        except Exception as e:
            return _json_response({"ok": False, "error": str(e)}, status=500)

    async def _banana_automation_history_clear_post(request):
        try:
            return _json_response(_clear_automation_history())
        except Exception as e:
            return _json_response({"ok": False, "error": str(e)}, status=500)

    _safe_add_route("GET", MANIFEST_ROUTE, _banana_ecommerce_manifest_get)
    _safe_add_route("GET", CONFIG_ROUTE, _banana_ecommerce_prompt_config_get)
    _safe_add_route("POST", CONFIG_ROUTE, _banana_ecommerce_prompt_config_post)
    _safe_add_route("GET", CONFIG_DEFAULTS_ROUTE, _banana_ecommerce_prompt_config_defaults)

    # 主面板路由 + 老面板路由兜底。这样旧 JS 缓存或旧按钮也能打开“冬之韵”。
    for editor_path in [EDITOR_ROUTE, "/banana/prompt-editor", "/banana/ecommerce-editor", "/ai-ecommerce/editor"]:
        _safe_add_route("GET", editor_path, _banana_triple_view_editor)

    _safe_add_route("GET", RUNTIME_ROUTE, _banana_ecommerce_runtime_results_get)
    _safe_add_route("POST", RUNTIME_CLEAR_ROUTE, _banana_ecommerce_runtime_clear_post)
    _safe_add_route("POST", AUTOMATION_SELECT_FOLDER_ROUTE, _banana_automation_select_folder_post)
    _safe_add_route("POST", AUTOMATION_PREVIEW_ROUTE, _banana_automation_preview_post)
    _safe_add_route("GET", AUTOMATION_HISTORY_ROUTE, _banana_automation_history_get)
    _safe_add_route("POST", AUTOMATION_HISTORY_CLEAR_ROUTE, _banana_automation_history_clear_post)

    logger.success(f"Banana 图像生成路由检查完成，冬之韵面板主路由: {EDITOR_ROUTE}")


class BananaPanelThreeViewNode:
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("front_image", "side_image", "back_image", "images", "info", "mp4url")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "AI电商/提示词模板"

    @classmethod
    def INPUT_TYPES(cls):
        cfg = _ecommerce_manifest()

        mode_options = list(_manifest_mode_options().keys())
        if not mode_options:
            mode_options = list(MODE_OPTIONS.keys())

        model_options = _enum_source_options("model_map", ["banano2", "banano-pro", "gemini3.1-pro"])
        image_size_options = _enum_source_options("image_size_options", ["1K", "2K", "4K", "8K（默认16:9）"])
        aspect_options = _enum_source_options(
            "aspect_ratio_options",
            ["Auto", "1:1 (方形)", "3:4 (竖屏标准)", "9:16 (竖屏/手机)", "16:9 (横屏宽幅)"],
        )

        default_mode = str(
            cfg.get("default_prompt_template")
            or cfg.get("default_mode")
            or "动作迁移"
        )
        if default_mode not in mode_options:
            default_mode = mode_options[0]

        default_model = _enum_source_display(
            "model_map",
            cfg.get("default_model") or _manual_model_default(),
            "banano2",
        )
        if default_model not in model_options:
            default_model = model_options[0]

        default_size = _enum_source_display(
            "image_size_options",
            cfg.get("default_image_size") or _manual_image_size_default(),
            "4K",
        )
        if default_size not in image_size_options:
            default_size = "4K" if "4K" in image_size_options else image_size_options[0]

        default_ratio = _enum_source_display(
            "aspect_ratio_options",
            cfg.get("default_aspect_ratio") or _manual_aspect_ratio_default("16:9"),
            "16:9 (横屏宽幅)",
        )
        if default_ratio not in aspect_options:
            default_ratio = "16:9 (横屏宽幅)" if "16:9 (横屏宽幅)" in aspect_options else aspect_options[0]

        required = {
            "api_key": (
                "STRING",
                {
                    "default": _cfg("api_key", ""),
                    "multiline": False,
                    "tooltip": "填入 API Key；留空时尝试读取 config.ini 的 api_key",
                },
            ),
            "mode": (
                mode_options,
                {
                    "default": default_mode,
                    "tooltip": "提示词模板。注意：这里不是大模型 model，前端同步只会改这个字段，不会修改 model。",
                },
            ),
            "model": (
                model_options,
                {
                    "default": default_model,
                    "tooltip": "大模型 model。提示词模板同步不会修改这个字段。",
                },
            ),
            "image_size": (
                image_size_options,
                {
                    "default": default_size,
                    "tooltip": "三张图使用同一尺寸。",
                },
            ),
            "aspect_ratio": (
                aspect_options,
                {
                    "default": default_ratio,
                    "tooltip": "三张图使用同一宽高比。",
                },
            ),
            "generate_scope": (
                _THREE_VIEW_SCOPE_OPTIONS,
                {
                    "default": "全部并发生成",
                    "tooltip": "质量不满意时可只重新生成某一个视图；其他视图会使用本节点上一次成功缓存结果。",
                },
            ),
            "auto_retry_until_success": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "开启后，单个视图失败或不出图会自动重试，直到成功或达到最大重试次数。",
                },
            ),
            "max_retry_per_view": (
                "INT",
                {
                    "default": 8,
                    "min": 1,
                    "max": 999,
                    "step": 1,
                    "tooltip": "每个视图最多自动重试次数。建议 5-12；填太大会导致节点运行很久。",
                },
            ),
            "retry_interval_sec": (
                "FLOAT",
                {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 30.0,
                    "step": 0.1,
                    "tooltip": "单路失败后的重试间隔秒数。",
                },
            ),
        }

        optional = {
            "mode_actual": (
                "STRING",
                {
                    "default": "",
                    "multiline": False,
                    "tooltip": "前端同步用的模板内部 key；通常留空。",
                },
            ),
            "cache_key": (
                "STRING",
                {
                    "default": "",
                    "multiline": False,
                    "tooltip": "可选缓存 key；留空则按当前节点 ID 和模板隔离。",
                },
            ),
            "labels_prefix": (
                "STRING",
                {
                    "default": "",
                    "multiline": False,
                    "tooltip": "可选输出标题前缀；留空自动使用模板名。",
                },
            ),
            "automation_payload": (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "tooltip": "自动化文件夹映射 JSON。由右下角自动化面板写入；不影响普通单次生成。",
                },
            ),
        }

        slot_count = int(cfg.get("optional_image_slots") or _NODE.get("optional_image_slots", 10) or 10)
        for i in range(1, slot_count + 1):
            optional[f"image_{i}"] = (
                "IMAGE",
                {
                    "tooltip": f"参考图 {i}；同一批上传图会复用到正面/侧面/背面三个并发请求",
                },
            )

        return {
            "required": required,
            "optional": optional,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def generate(
        self,
        api_key: str,
        mode: str,
        model: str,
        image_size: str,
        aspect_ratio: str,
        generate_scope: str = "全部并发生成",
        auto_retry_until_success: bool = True,
        max_retry_per_view: int = 8,
        retry_interval_sec: float = 1.5,
        mode_actual: str = "",
        cache_key: str = "",
        labels_prefix: str = "",
        automation_payload: str = "",
        unique_id=None,
        **kwargs,
    ):
        start = time.time()
        resolved_key = str(api_key or "").strip() or _cfg("api_key", "")

        if not resolved_key:
            msg = "请在节点中填入 API Key"
            logger.error(msg)
            img = _error_img(msg)
            return _return_images_with_ui_preview((img, img, img, img, msg, ""), label="banana_panel_error")

        template_actual = _mode_actual_from_display(mode, mode_actual)
        template_display = _mode_display_from_actual(template_actual)

        if template_actual in DELETED_MODE_KEYS:
            msg = f"提示词模板 {template_actual} 已删除，请重新选择模板"
            logger.error(msg)
            img = _error_img(msg)
            return _return_images_with_ui_preview((img, img, img, img, msg, ""), label="banana_panel_error")

        if _automation_enabled(automation_payload):
            return self.generate_automation(
                resolved_key=resolved_key,
                mode=mode,
                model=model,
                image_size=image_size,
                aspect_ratio=aspect_ratio,
                auto_retry_until_success=auto_retry_until_success,
                max_retry_per_view=max_retry_per_view,
                retry_interval_sec=retry_interval_sec,
                generate_scope=generate_scope,
                mode_actual=mode_actual,
                cache_key=cache_key,
                labels_prefix=labels_prefix,
                automation_payload=automation_payload,
                unique_id=unique_id,
            )

        automation_info = ""
        if str(automation_payload or "").strip():
            try:
                automation_data = json.loads(str(automation_payload))
                groups = automation_data.get("preview_groups") or automation_data.get("groups") or []
                automation_info = f"automation_payload: 已填写但 enabled=false；预览组数={len(groups)}"
            except Exception:
                automation_info = "automation_payload: 已填写，但不是有效 JSON"

        try:
            image_urls = _upload_reference_images_for_node(kwargs, resolved_key)
        except Exception as e:
            msg = f"参考图上传失败: {e}"
            logger.error(msg)
            img = _error_img(msg)
            return _return_images_with_ui_preview((img, img, img, img, msg, ""), label="banana_panel_error")

        prompts = {
            "front": _compose_prompt(template_actual, "front"),
            "side": _compose_prompt(template_actual, "side"),
            "back": _compose_prompt(template_actual, "back"),
        }

        labels = str(labels_prefix or template_display or template_actual).strip()
        run_cache_key = str(cache_key or "").strip() or f"banana_image_generation:{unique_id}:{template_actual}"

        logger.info(
            f"Banana 图像生成开始: mode={template_display}/{template_actual}, model={model}, "
            f"size={image_size}, ratio={aspect_ratio}, scope={generate_scope}, ref_image_count={len(image_urls)}"
        )

        try:
            result = _run_three_view_jobs(
                api_key=resolved_key,
                model=model,
                image_size=image_size,
                aspect_ratio=aspect_ratio,
                image_urls=image_urls,
                prompts=prompts,
                labels_prefix=f"{labels}-" if labels else "",
                generate_scope=generate_scope,
                cache_key=run_cache_key,
                auto_retry_until_success=auto_retry_until_success,
                max_retry_per_view=max_retry_per_view,
                retry_interval_sec=retry_interval_sec,
            )
        except Exception as e:
            msg = str(e)[:2500]
            logger.error(f"Banana 图像生成失败: {msg}")
            img = _error_img("Banana 图像生成失败")
            return _return_images_with_ui_preview((img, img, img, img, msg, ""), label="banana_panel_error")

        elapsed = time.time() - start
        ordered = result["ordered"]

        lines = [
            f"✅ Banana 图像生成完成，耗时 {elapsed:.1f}s",
            f"theme: 冬之韵 / Winter Rhyme",
            f"mode: {template_display} ({template_actual})",
            f"model: {model}",
            f"image_size: {image_size}",
            f"aspect_ratio: {aspect_ratio}",
            f"generate_scope: {result.get('generate_scope')}",
            f"auto_retry_until_success: {result.get('auto_retry_until_success')}",
            f"max_retry_per_view: {result.get('max_retry_per_view')}",
            f"retry_interval_sec: {result.get('retry_interval_sec')}",
            f"cache_key: {result.get('cache_key')}",
            f"ref_image_count: {len(image_urls)}",
            "prompt_source: frontend_panel_config",
            f"prompt_config: {os.path.basename(_prompt_config_path())}",
            f"editor: {EDITOR_ROUTE}",
            "字段隔离: mode=提示词模板；model=大模型；前端同步不会覆盖 model",
            "输出接口: front_image=正面图, side_image=侧面图, back_image=背面图, images=三张批量合集",
            "输出顺序: images[0]=正面图, images[1]=侧面图, images[2]=背面图",
        ]

        if automation_info:
            lines.append(automation_info)

        for idx, item in enumerate(ordered, start=1):
            lines.append(
                f"{idx}. {item.get('label', '')} | 耗时 {float(item.get('elapsed') or 0):.1f}s | seed={item.get('seed', '')} | "
                f"size={item.get('image_size', '')} | ratio={item.get('aspect_ratio', '')}"
            )
            if str(item.get("info") or "").strip():
                lines.append(str(item["info"]))

        summary = "\n".join(lines)

        logger.summary("Banana 图像生成完成", {
            "输出": "正面/侧面/背面 + batch",
            "耗时": f"{elapsed:.1f}s",
            "主题": "冬之韵",
            "模板": f"{template_display}/{template_actual}",
            "大模型model": model,
            "尺寸": image_size,
            "宽高比": aspect_ratio,
            "生成范围": result.get("generate_scope"),
            "缓存Key": result.get("cache_key"),
            "失败视图": ",".join((result.get("errors_by_key") or {}).keys()) or "无",
            "ref_image_count": len(image_urls),
        })

        return _return_images_with_ui_preview((
            result["front"],
            result["side"],
            result["back"],
            result["batch"],
            summary,
            "",
        ), label="banana_panel_three_view")


    def generate_automation(
        self,
        *,
        resolved_key: str,
        mode: str,
        model: str,
        image_size: str,
        aspect_ratio: str,
        auto_retry_until_success: bool,
        max_retry_per_view: int,
        retry_interval_sec: float,
        generate_scope: str = "全部并发生成",
        mode_actual: str = "",
        cache_key: str = "",
        labels_prefix: str = "",
        automation_payload: str = "",
        unique_id=None,
    ):
        start = time.time()
        cfg = _normalize_automation_payload(automation_payload)

        template_actual = _mode_actual_from_display(mode, mode_actual)
        template_display = _mode_display_from_actual(template_actual)
        labels = str(labels_prefix or template_display or template_actual).strip()
        run_cache_key = str(cache_key or "").strip() or f"banana_image_automation:{unique_id}:{template_actual}"

        if not cfg.get("input_roots"):
            msg = "自动化已启用，但没有 input_roots。请在自动化面板选择输入根目录。"
            img = _error_img(msg)
            return _return_images_with_ui_preview((img, img, img, img, msg, ""), label="banana_panel_error")
        if not cfg.get("output_root"):
            msg = "自动化已启用，但没有 output_root。请在自动化面板选择输出根目录。"
            img = _error_img(msg)
            return _return_images_with_ui_preview((img, img, img, img, msg, ""), label="banana_panel_error")

        groups = _build_sequence_groups(
            cfg["input_roots"],
            output_root=cfg["output_root"],
            require_all_roots_present=bool(cfg.get("require_all_roots_present")),
        )
        all_group_count = len(groups)
        run_sequences = set(str(x) for x in (cfg.get("run_sequences") or []) if str(x).strip())
        if run_sequences:
            groups = [g for g in groups if str(g.get("sequence") or "") in run_sequences]
        if not groups:
            if run_sequences:
                msg = f"自动化没有找到指定序号组：{', '.join(sorted(run_sequences))}。请先在自动化面板预览分组，确认序号存在。"
            else:
                msg = "自动化没有扫描到任何有效序号组。请确认每个输入根目录下直接放置带数字序号的图片，例如 001.png、002.png。"
            img = _error_img(msg)
            return _return_images_with_ui_preview((img, img, img, img, msg, ""), label="banana_panel_error")

        prompts = {
            "front": _compose_prompt(template_actual, "front"),
            "side": _compose_prompt(template_actual, "side"),
            "back": _compose_prompt(template_actual, "back"),
        }

        group_concurrency = int(cfg.get("group_concurrency") or 3)
        logger.info(
            f"Banana 自动化开始: groups={len(groups)}, concurrency={group_concurrency}, "
            f"mode={template_display}/{template_actual}, model={model}, output_root={cfg['output_root']}"
        )

        results: List[Dict[str, Any]] = []
        futures = []
        with ThreadPoolExecutor(max_workers=max(1, min(10, group_concurrency))) as executor:
            for group in groups:
                futures.append(executor.submit(
                    _run_automation_one_group,
                    group=group,
                    cfg=cfg,
                    api_key=resolved_key,
                    model=model,
                    image_size=image_size,
                    aspect_ratio=aspect_ratio,
                    prompts=prompts,
                    labels=labels,
                    cache_key=run_cache_key,
                    auto_retry_until_success=auto_retry_until_success,
                    max_retry_per_view=max_retry_per_view,
                    retry_interval_sec=retry_interval_sec,
                    generate_scope=generate_scope,
                ))

            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda x: _sequence_sort_key(str(x.get("sequence") or "")))
        ok_results = [r for r in results if r.get("ok")]
        fail_results = [r for r in results if not r.get("ok")]
        elapsed = time.time() - start

        representative = ok_results[-1] if ok_results else None
        if representative:
            front = representative.get("front")
            side = representative.get("side")
            back = representative.get("back")
            batch = representative.get("batch")
            if batch is None:
                batch = _cat_image_batches_safe([front, side, back])
        else:
            err_text = "自动化全部失败"
            front = side = back = batch = _error_img(err_text)

        lines = [
            f"✅ Banana 自动化批处理完成，耗时 {elapsed:.1f}s",
            f"mode: {template_display} ({template_actual})",
            f"model: {model}",
            f"image_size: {image_size}",
            f"aspect_ratio: {aspect_ratio}",
            f"input_roots: {len(cfg['input_roots'])}",
            f"groups: {len(groups)} / all_groups: {all_group_count}",
            f"run_sequences: {', '.join(sorted(run_sequences)) if run_sequences else '全部'}",
            f"generate_scope: {generate_scope}",
            f"success: {len(ok_results)}",
            f"failed: {len(fail_results)}",
            f"group_concurrency: {group_concurrency}",
            f"max_images_per_group: {cfg.get('max_images_per_group')}",
            f"output_root: {cfg['output_root']}",
            "输入规则: 只扫描输入根目录下的直接图片文件，例如 input_root_01/001.png；输出目录规则: output_序号/run_01/，图片文件 front.png / side.png / back.png",
        ]

        if cfg.get("save_video"):
            lines.append("提示: 当前图像自动化节点本身不产生视频；视频已拆分为独立【🍌 Banana｜生视频】节点。")

        for r in results:
            if r.get("ok"):
                lines.append(f"✅ {r.get('sequence')} -> {r.get('output_dir')} | 输入图片 {r.get('input_image_count')} 张")
            else:
                lines.append(f"❌ {r.get('sequence')} -> {r.get('output_dir')} | {r.get('error')}")

        summary = "\n".join(lines)
        logger.summary("Banana 自动化批处理完成", {
            "总组数": len(groups),
            "成功": len(ok_results),
            "失败": len(fail_results),
            "耗时": f"{elapsed:.1f}s",
            "并发": group_concurrency,
            "输出根目录": cfg["output_root"],
        })

        return _return_images_with_ui_preview((front, side, back, batch, summary, ""), label="banana_panel_automation")


_register_ecommerce_routes()

NODE_CLASS_MAPPINGS = {
    "BananaPanelThreeViewNode": BananaPanelThreeViewNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BananaPanelThreeViewNode": "🍌 Banana｜图像生成",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "BananaPanelThreeViewNode",
    "_ecommerce_manifest",
    "_manifest_mode_options",
    "_read_prompt_config",
    "_save_prompt_config",
    "_default_prompt_config",
]
