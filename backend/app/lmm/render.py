"""框绘制：结果视频（可下载的 mp4）与**告警证据图**共用同一份实现。

★前端已经不画框了★（原来的 `components/lmm-canvas.tsx` 已删除）：告警页上
画面必须是纯视频，框只出现在**服务端合成的证据图**里 —— 因为证据图是"过去那一帧
的原图 + 那一帧 VLX 给的框"，前端既拿不到那一帧的像素，也不该拿坐标去重绘
（重绘出来的框与图容易错位，那就不是证据了）。

这里因此只剩下两件事：

  * `draw_boxes` —— 给结果视频与证据图**画框 + 中文标签**；
  * `draw_evidence_jpeg` —— 把一张 JPEG 解码 → 画框 → 再编码，供证据图使用。

样式约定：
  * 调色板按 label 哈希取色（同一个哈希算法，保证同类同色、跨帧稳定）；
  * 框内 18% 同色填充 + 实线描边，线宽 / 字号按 `max(宽/640, 1)` 缩放；
  * 框上方单行标签「疑似 xxx」：**不画数字、不画状态**（置信度对用户不可解释）；
  * `rejected` / `error` 的框灰化且不填充。

中文标签必须走 PIL：OpenCV 的 `putText` 画不出汉字，只会得到一串 "???"。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger(__name__)

# ── ★证据图的画质预算★ ───────────────────────────────────────────────────────
#: ★为什么证据图必须给到 95★：证据图是要**放大看远距离目标**的。源文件本身是
#: 有损 JPEG，画框又必须「解码 → 画 → 再编码」一次，等于二次压缩；质量给低了，
#: 远距离小目标放大后只剩一片马赛克，证据就失去意义了。一次告警三张图，体积
#: 从 ~180 KB 涨到 ~600 KB 完全可接受（它不进 WS、不逐帧传）。
EVIDENCE_JPEG_QUALITY = 95

#: ★证据展示图的编码格式★：`png`（默认，无损）/ `jpg`（有损）。
#: 由环境变量 `LMM_EVIDENCE_FORMAT` 决定：`png` | `jpg98` | `jpg95`。
#: 依据见 `.workbuddy/_encode_mode_experiment.txt`：PNG 1.8 MB/155 ms，
#: JPEG q98 0.43 MB/7.5 ms —— 展示图默认走 PNG，第二代压缩彻底为零。
EVIDENCE_IMAGE_FORMAT = (os.environ.get("LMM_EVIDENCE_FORMAT") or "png").lower()
if EVIDENCE_IMAGE_FORMAT in ("jpg98", "jpeg98"):
    EVIDENCE_IMAGE_FORMAT, EVIDENCE_JPEG_QUALITY = "jpg", 98
elif EVIDENCE_IMAGE_FORMAT in ("jpg95", "jpeg95"):
    EVIDENCE_IMAGE_FORMAT, EVIDENCE_JPEG_QUALITY = "jpg", 95
elif EVIDENCE_IMAGE_FORMAT in ("jpg", "jpeg"):
    EVIDENCE_IMAGE_FORMAT = "jpg"
else:
    EVIDENCE_IMAGE_FORMAT = "png"
EVIDENCE_IMAGE_EXT = "png" if EVIDENCE_IMAGE_FORMAT == "png" else "jpg"

#: 列表缩略图（`001_thumb.jpg`）：只给右侧卡片的小图用，**弹窗绝不用它**。
#: 卡片上一张 1920×1080 原图只是为了显示 96 px 宽，纯浪费流量。
THUMB_MAX_SIDE = 320
THUMB_JPEG_QUALITY = 70

#: ★清晰度诊断里"排除标注区域"的外扩像素★。
#: 实测（57 张 1920×1080，真实 RTSP）：
#:   pad=48  → ratio 最高 1.28（标签文字画在框**上方偏右**，48 px 盖不住）；
#:   pad=120 → ratio ∈ [0.9996, 1.0001]，另有 2 张在 x 方向超出 4 px；
#:   pad=160 → 框外**逐像素最大差 = 0**（57/57），ratio 精确 = 1.0000。
#: 取 160：既完整盖住"框线 + 标签"，又不至于把整幅画面排除掉。
SHARPNESS_EXCLUDE_PAD = 160

#: 类别 → 颜色。顺序固定：调色板一变，所有历史证据图的配色就对不上了。
_PALETTE_HEX = (
    "#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6",
    "#3b82f6", "#6366f1", "#a855f7", "#ec4899",
)

#: 灰化描边（对应前端 rejected/error 的 #9ca3af）
_GREY_HEX = "#9ca3af"

#: 中文字体候选。视频标签是中文，找不到中文字体就只能画框不画字（不报错）。
_CJK_FONTS = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyh.ttf",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/Deng.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)

_FONT_PATH: Optional[str] = None
_FONT_PATH_RESOLVED = False
_FONT_CACHE: Dict[int, Any] = {}


# ── 颜色 / 文案（与前端同源） ────────────────────────────────────────────────


def _hex_to_bgr(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def color_of(label: str) -> Tuple[int, int, int]:
    """按 label 取色 —— 哈希算法必须与 JS 版一致。

    JS 里是 `h = (h * 31 + label.charCodeAt(i)) >>> 0`，Python 侧 `& 0xFFFFFFFF`
    等价于 `>>> 0`（汉字都在 BMP，`charCodeAt` 即码点）。
    """
    h = 0
    for ch in label or "":
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return _hex_to_bgr(_PALETTE_HEX[h % len(_PALETTE_HEX)])


def label_text(box: Dict[str, Any]) -> str:
    """框上文案，规则与前端 `labelText()` 相同：统一「疑似 xxx」，已带前缀不重复加。"""
    name = str(box.get("display_name") or box.get("label_zh") or box.get("label") or "").strip()
    if not name:
        return "疑似目标"
    return name if name.startswith("疑似") else f"疑似{name}"


# ── 字体 ─────────────────────────────────────────────────────────────────────


def _font_path() -> Optional[str]:
    global _FONT_PATH, _FONT_PATH_RESOLVED
    if not _FONT_PATH_RESOLVED:
        _FONT_PATH_RESOLVED = True
        for candidate in _CJK_FONTS:
            if Path(candidate).is_file():
                _FONT_PATH = candidate
                break
        if _FONT_PATH is None:
            LOG.warning("[LMM] 未找到中文字体，结果视频只画框不写标签")
    return _FONT_PATH


def _font(size: int):
    try:
        from PIL import ImageFont
    except Exception:  # noqa: BLE001
        return None
    path = _font_path()
    if path is None:
        return None
    size = max(8, int(size))
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            LOG.debug("[LMM] 字体加载失败：%s", path, exc_info=True)
            _FONT_CACHE[size] = None
    return _FONT_CACHE[size]


# ── 绘制 ─────────────────────────────────────────────────────────────────────


def _clip_box(box: Dict[str, Any], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    try:
        x1 = int(round(float(box.get("x1", 0))))
        y1 = int(round(float(box.get("y1", 0))))
        x2 = int(round(float(box.get("x2", 0))))
        y2 = int(round(float(box.get("y2", 0))))
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0, min(x1, width - 1)), max(0, min(x2, width - 1))))
    y1, y2 = sorted((max(0, min(y1, height - 1)), max(0, min(y2, height - 1))))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2


def _put_label(frame, text: str, x: int, y: int, color, scale: float) -> None:
    """在框上方画单行标签。用 PIL 画中文，画不成（没字体/没 Pillow）就安静跳过。"""
    font = _font(round(12 * scale))
    if font is None or not text:
        return
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except Exception:  # noqa: BLE001
        return
    pad_x = int(round(5 * scale))
    pad_y = int(round(3 * scale))
    try:
        left, top, right, bottom = font.getbbox(text)
    except Exception:  # noqa: BLE001
        return
    text_w = max(1, right - left)
    text_h = max(1, bottom - top)
    box_w = text_w + pad_x * 2
    box_h = text_h + pad_y * 2
    height, width = frame.shape[:2]
    ty = y - box_h - 2 if y - box_h - 2 > 0 else y + 2
    tx = min(max(0, x), max(0, width - box_w))
    ty = min(max(0, ty), max(0, height - box_h))
    box_w = min(box_w, width - tx)
    box_h = min(box_h, height - ty)
    if box_w < 2 or box_h < 2:
        return
    r, g, b = color[2], color[1], color[0]
    patch = Image.new("RGB", (box_w, box_h), (r, g, b))
    draw = ImageDraw.Draw(patch)
    draw.text((pad_x - left, pad_y - top), text, font=font, fill=(255, 255, 255))
    frame[ty:ty + box_h, tx:tx + box_w] = np.asarray(patch)[:, :, ::-1]


def draw_boxes(frame, boxes: List[Dict[str, Any]], fill_alpha: float = 0.18):
    """就地把 `boxes` 画到 `frame`（BGR ndarray）上，返回同一个 frame。

    `boxes` 用 `RuleEngine.decorate()` 的输出即可：只读 x1/y1/x2/y2、label*、vlm_state。
    """
    if frame is None or not boxes:
        return frame
    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001
        return frame

    height, width = frame.shape[:2]
    scale = max(width / 640.0, 1.0)
    line_w = max(1, int(round(1.6 * scale)))

    for box in boxes:
        rect = _clip_box(box, width, height)
        if rect is None:
            continue
        x1, y1, x2, y2 = rect
        state = str(box.get("vlm_state") or "")
        rejected = state in ("rejected", "error")
        color = _hex_to_bgr(_GREY_HEX) if rejected else color_of(str(box.get("label") or ""))

        if not rejected:
            # 半透明填充：只混这一小块 ROI，整帧不做运算
            roi = frame[y1:y2, x1:x2]
            overlay = np.empty_like(roi)
            overlay[:] = color
            cv2.addWeighted(overlay, fill_alpha, roi, 1.0 - fill_alpha, 0, dst=roi)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_w)
        _put_label(frame, label_text(box), x1, y1, color, scale)

    return frame


def jpeg_wh(data: bytes) -> Tuple[int, int]:
    """只读 JPEG 头拿宽高（SOF 标记），**不解码整图** —— 落盘后要把它写进事件，
    前端弹窗底部显示「1920 × 1080」；为这点信息解码 6 MB 不值当。
    """
    try:
        i, n = 2, len(data)
        while i < n - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seglen = (data[i + 2] << 8) | data[i + 3]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h = (data[i + 5] << 8) | data[i + 6]
                w = (data[i + 7] << 8) | data[i + 8]
                return w, h
            i += 2 + seglen
    except Exception:  # noqa: BLE001
        pass
    return 0, 0


def encode_jpeg_bytes(frame: Any, quality: int = EVIDENCE_JPEG_QUALITY
                      ) -> Optional[bytes]:
    """整帧 → JPEG 字节，**不缩放**（给证据图用，与预览通道的 `encode_jpeg_b64`
    完全相反：那边为了 WS 体积必须缩，这边为了证据必须保真）。
    """
    if frame is None:
        return None
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return None
    try:
        ok, enc = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return enc.tobytes()
    except Exception:  # noqa: BLE001
        LOG.warning("[LMM] 整帧 JPEG 编码失败", exc_info=True)
        return None


def encode_image_bytes(frame: Any, fmt: str = EVIDENCE_IMAGE_FORMAT,
                       quality: int = EVIDENCE_JPEG_QUALITY
                       ) -> Optional[bytes]:
    """★证据展示图编码★：`png`（无损，默认）/ `jpg`（有损，`quality` 生效）。

    ★为什么默认 PNG★（实测，见 `.workbuddy/_encode_mode_experiment.txt`）：
      source(q95) → 解码 → 画框 → 再编码，这一步如果还是 JPEG，就是**第二代有损**。
      实测同一批真实帧：PNG 1.8 MB / 155 ms，JPEG q98 0.43 MB / 7.5 ms，JPEG q95
      0.33 MB / 8 ms —— PNG 贵 4.3 倍体积 + 20 倍时间，换来的是**真正零二次损失**。
      证据图的唯一用途是被放大看，且现在写入已挪到后台 worker（不阻塞检测），
      所以按"质量稳定优先"默认选 PNG；想省磁盘用 `LMM_EVIDENCE_FORMAT=jpg98`。
    """
    if frame is None:
        return None
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return None
    try:
        if str(fmt).lower() in ("png", ".png", "image/png"):
            ok, enc = cv2.imencode(".png", frame,
                                   [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        else:
            ok, enc = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return enc.tobytes()
    except Exception:  # noqa: BLE001
        LOG.warning("[LMM] 证据图编码失败", exc_info=True)
        return None


def sharpness(frame: Any, exclude_box: "Optional[Sequence[float]]" = None,
              pad: int = SHARPNESS_EXCLUDE_PAD) -> float:
    """★清晰度★ = 灰度 Laplacian 方差（越大越锐），用于**诊断**而非语义判定。

    为什么用它：只依赖 numpy/cv2（不引新模型），一条告警三张图各算一次的成本
    ~10 ms（在后台 worker 里，不影响检测）。

    ★它的局限必须知道★：Laplacian 方差对"量化噪声/块效应"不敏感 —— 实测
    q95→q95 二次压缩的 ratio 是 0.999（看不出损失），但像素确实又损了一次。
    所以它主要用来区分**源帧本身糊**（ratio 接近 1 但绝对值低）与
    **处理链把图弄糊**（ratio 明显 < 1）。

    ★★ `exclude_box`：必须排除画上去的框★★
    第一版没排除，结果源帧很糊的那些图 ratio 居然 1.9~2.0 —— 不是"变清晰了"，
    而是**红框和标签文字本身就是高频边缘**，在一张本来就糊的图上加几条锐利直线，
    Laplacian 方差直接翻倍。这会让人误判"处理链动手脚了"。
    所以 source 和 evidence **两边都排除同一块区域**（框外扩 `pad` 像素，
    把框线自身的邻域也排掉），比值才只反映编码/缩放的真实损失。
    """
    if frame is None:
        return 0.0
    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001
        return 0.0
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:  # noqa: BLE001
        return 0.0
    try:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        box = list(exclude_box or [])
        if len(box) >= 4:
            h, w = gray.shape[:2]
            x1 = max(0, int(min(box[0], box[2])) - pad)
            y1 = max(0, int(min(box[1], box[3])) - pad)
            x2 = min(w, int(max(box[0], box[2])) + pad)
            y2 = min(h, int(max(box[1], box[3])) + pad)
            if x2 > x1 and y2 > y1:
                mask = np.ones(lap.shape, dtype=bool)
                mask[y1:y2, x1:x2] = False
                vals = lap[mask]
                if vals.size > 1000:
                    return float(vals.var())
        return float(lap.var())
    except Exception:  # noqa: BLE001
        return 0.0


def make_thumbnail_jpeg(data: bytes, max_side: int = THUMB_MAX_SIDE,
                        quality: int = THUMB_JPEG_QUALITY) -> Optional[bytes]:
    """证据图 → 列表缩略图字节（**只供卡片小图**，弹窗一律用原图）。

    失败返回 None（宁可让卡片去拉原图，也不要因为缩略图没生成就丢证据）。
    """
    if not data:
        return None
    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001
        return None
    try:
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        side = max(w, h)
        if side > max_side:
            scale = max_side / float(side)
            frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return enc.tobytes()
    except Exception:  # noqa: BLE001
        LOG.debug("[LMM] 缩略图生成失败", exc_info=True)
        return None


def draw_evidence_jpeg(data: bytes, box: Optional[Dict[str, Any]],
                       quality: int = EVIDENCE_JPEG_QUALITY) -> bytes:
    """★告警证据图★：把**这一帧 VLX 自己给的那个框**画进 JPEG，返回新的 JPEG 字节。

    为什么必须**在服务端**画（而不是前端拿 bbox 去 canvas 重绘）：

      1. 证据图是**证据**。前端重绘意味着"这张图上画的框"是前端算出来的，不是
         VLX 说的 —— 一旦框与图错位（缩放/裁剪/letterbox 都会错位），证据就假了；
      2. 证据帧是**过去**的帧（ring 里捞回来的），前端根本拿不到那一帧的原图，
         只能拿当前画面顶替 —— 那就更不是证据了。

    所以：图是那一帧的原图，框是那一帧 VLX 的原始 xyxy，标签是 VLX 的类别名，
    三者在服务端一次性合成，前端只负责展示。

    任何一步失败都**原样返回输入**：宁可给一张没框的图，也不丢图。
    """
    if not data or not box:
        return data
    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001
        return data
    try:
        buf = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return data
        draw_boxes(frame, [box], fill_alpha=0.10)
        ok, enc = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return data
        return enc.tobytes()
    except Exception:  # noqa: BLE001
        LOG.warning("[LMM] 证据图画框失败，退回原图", exc_info=True)
        return data


def encode_jpeg_b64(frame: Any, max_side: int = 960, quality: int = 72) -> Optional[str]:
    """帧 → JPEG base64（缩到 `max_side` 以内，避免 WS 消息过大）。

    放在 render.py 是为了让 **legacy 实时推送** 与 **R1 预览通道** 共用同一份实现
    —— 两处各写一份，改了缩略尺寸只会改一半，画面大小对不上很难查。
    """
    if frame is None:
        return None
    try:
        import base64
        import io

        import numpy as np  # noqa: F401
        from PIL import Image
    except Exception:  # noqa: BLE001
        return None
    try:
        img = Image.fromarray(frame[:, :, ::-1])
        side = max(img.size)
        if side > max_side:
            scale = max_side / float(side)
            img = img.resize((max(1, int(img.size[0] * scale)),
                              max(1, int(img.size[1] * scale))))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(quality))
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        LOG.debug("[LMM] 帧编码失败", exc_info=True)
        return None


def output_dir() -> Path:
    """结果视频 / 检出数据的落盘目录（与推理页既有约定一致）。

    放 `workspace/temp/inference/`：启动清理器**刻意保留**该目录，而且 `*_input.mp4`
    本来就在这儿（LMM 页的「最近视频」按它列举），输入与输出放一起最好找。
    """
    from config import settings

    path = Path(settings.TEMP_DIR) / "inference"
    path.mkdir(parents=True, exist_ok=True)
    return path
