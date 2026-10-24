"""Bounded, account-local media upload, shared by detection and the resource library."""

from pathlib import Path
import uuid
from fastapi import HTTPException

VIDEO_SUFFIXES = {
    ".mp4",
    ".m4v",
    ".mov",
    ".avi",
    ".mkv",
    ".flv",
    ".wmv",
    ".webm",
    ".mpg",
    ".mpeg",
    ".ts",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_MEDIA_BYTES = 512 * 1024 * 1024


def owned_file(source, root, must_exist=True):
    root = Path(root).resolve()
    path = Path(str(source)).expanduser().resolve()
    if not path.is_relative_to(root):
        raise HTTPException(
            403, "不能读取其他工作区的文件，请上传当前账号的图片或视频。"
        )
    if must_exist and not path.is_file():
        raise HTTPException(404, "输入文件不存在，请重新上传。")
    return path


async def save_upload(upload, directory, allowed, limit=MAX_MEDIA_BYTES):
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(422, "不支持的文件类型。")
    if suffix in IMAGE_SUFFIXES:
        limit = min(limit, 64 * 1024 * 1024)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (uuid.uuid4().hex + suffix)
    size = 0
    try:
        with path.open("xb") as output:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        413, f"文件不能超过 {limit // 1024 // 1024} MB。"
                    )
                output.write(chunk)
        if not size:
            raise HTTPException(400, "文件内容为空。")
        if suffix in IMAGE_SUFFIXES:
            from PIL import Image

            try:
                with Image.open(path) as image:
                    if image.width * image.height > 64_000_000:
                        raise HTTPException(413, "图片像素过大，请缩小后上传。")
                    image.verify()
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(422, "无法读取图片，请检查文件格式。") from None
        elif suffix in VIDEO_SUFFIXES:
            import asyncio

            def check_video():
                import cv2

                cap = cv2.VideoCapture(str(path))
                try:
                    return cap.isOpened() and cap.read()[0]
                finally:
                    cap.release()

            if not await asyncio.to_thread(check_video):
                raise HTTPException(422, "无法解码视频，请使用 H.264 MP4 等常用格式。")
        return path, size
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
