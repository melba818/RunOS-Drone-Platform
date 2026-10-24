'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Loader2, RotateCcw, X, ZoomIn, ZoomOut } from 'lucide-react'

/**
 * ★告警证据图查看器★：点告警卡片里唯一那张证据图开大图。
 *
 * ★★ 画质铁律（用户逐条拍定）★★
 *   · 中央大图加载的一定是 `original_url` —— **证据原图**，不缩放、不压缩、
 *     不 canvas 重绘、不重新截图。CSS 只改变显示尺寸，不动分辨率；
 *   · ★绝不 `original_url ?? thumbnail_url`★：原图还没 ready 就显示
 *     「证据图处理中…」，**不许先拿缩略图放大糊弄**——那是"看得见但认不出"；
 *   · 缩略图（320 px）只允许出现在**卡片**上，弹窗大图绝不用；
 *   · 底部显示「1920 × 1080 · 583 KB · PNG」，让人一眼确认"我看的确实是原图"。
 *
 * ★1-Hit + 单证据图（2026-09-26 产品定案）★
 *   一条告警 = 一张证据图 = 触发告警的那一次 VLX 检出帧。
 *   因此这里**没有**三图切换、左右箭头、底部缩略图条、1/3 计数、方向键切图 ——
 *   那些东西已经随"三帧确认"一起取消了。
 *   证据是后台 worker 异步产出的（pending → ready），所以对 `?v=<version>`
 *   与 `key` 的处理仍然必须保留：证据图是「同一个路径，内容后来才写进去」的，
 *   不带版本号浏览器会拿第一次的（可能是 404）缓存结果。
 *
 * 交互：
 *   · 大图 `max-height: 80vh` / `max-width: 90vw` / `object-fit: contain`
 *     —— **绝不能 cover**：cover 会把两侧裁掉，而目标很可能就在被裁掉的那一侧；
 *   · 滚轮缩放 0.5~8 倍（上限 8 是为了能看清远距离目标；≥2.5 倍时切成
 *     `pixelated` 渲染，放大看到的就是**原始像素**，不再被插值糊掉）；
 *   · 左键拖拽平移；双击复位；`Esc` 关闭。
 *
 * 这里**只负责显示**：证据图上的框与标签是后端用 VLX 原始 bbox 烧进像素的，
 * 前端不重绘、不叠任何覆盖层（前端重绘 = 画出来的框不再是 VLX 说的那个）。
 */

/** 一条证据图的完整信息（1-Hit：一条告警只有这一张） */
export interface LmmEvidenceImage {
  /** 证据在告警里的序号，恒为 0（保留字段，兼容旧接口） */
  index?: number
  /** `pending` / `processing` / `ready` / `error` */
  status?: string
  /** 证据帧号（**不展示**，只用于定位与验收比对 alert.frame_id） */
  frame_id?: number | null
  pts_ms?: number | null
  /** ★原图 URL★：弹窗只认它 */
  original_url: string
  /** 未画框的原始帧（StreamHub q95 字节原样落盘），仅供人工比对 */
  source_url?: string
  /** 标注证据失败时，弹窗仍以原始帧全分辨率展示。 */
  source_original_fallback?: boolean
  /** 320 px 缩略图 URL：**只给卡片用**，弹窗大图绝不用 */
  thumbnail_url?: string
  /** 该帧 VLX 给的 xyxy（**留档**，框已烧进像素，前端不画） */
  bbox?: number[]
  width?: number
  height?: number
  bytes?: number
  source_bytes?: number
  /** `png` / `jpg` —— 展示图编码格式 */
  format?: string
  /** 清晰度诊断（**只用于诊断**）：源帧 Laplacian 方差 */
  source_sharpness?: number
  sharpness?: number
  /** `sharpness / source_sharpness`：≈1 说明处理链没弄糊 */
  quality_ratio?: number
  /** 文件 mtime，用作缓存 busting 的版本号 */
  version?: string
}

interface Props {
  /** ★唯一那张证据图★（1-Hit：不再有数组） */
  image?: LmmEvidenceImage | null
  /** ★所属告警 id★：key 与 URL 都要带上它，避免不同告警的图片互相顶掉 */
  eventId?: string
  /** 顶部标题，一般用告警的目标描述 */
  title?: string
  /** 后端地址（证据图是后端路径，必须拼上它才能访问） */
  apiUrl?: string
  onClose: () => void
}

const MIN_ZOOM = 0.5
/** ★上限 8 倍★：远距离目标在 1080p 里往往只有几十像素，5 倍不够看清 */
const MAX_ZOOM = 8
/** 超过这个倍率改用最近邻渲染 —— 放大看的是**原始像素**，不是插值糊出来的 */
const PIXELATED_AT = 2.5

function humanKB(n?: number): string {
  if (!n || n <= 0) return ''
  if (n >= 1024 * 1024) return `${(n / 1048576).toFixed(1)} MB`
  return `${Math.round(n / 1024)} KB`
}

/**
 * ★缓存 busting★：URL 必须带版本号。
 * 证据图是「同一个路径，内容后来才写进去」的——不带 `?v=` 时浏览器会拿
 * 第一次的（可能是 404 / 旧图）缓存结果，表现为"图一直不出来"。
 */
function withVersion(base: string, url: string, version?: string): string {
  if (!url) return ''
  const u = `${base}${url}`
  if (!version) return u
  return u.includes('?') ? `${u}&v=${version}` : `${u}?v=${version}`
}

export function LmmAlertModal({ image, eventId, title, apiUrl, onClose }: Props) {
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null)
  const dragRef = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null)
  const stageRef = useRef<HTMLDivElement | null>(null)

  const reset = useCallback(() => {
    setZoom(1)
    setOffset({ x: 0, y: 0 })
  }, [])

  const clampZoom = useCallback((z: number) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z)), [])

  // 换一条告警（或证据从 pending 变 ready）时复位缩放，避免带着上一条的位移
  useEffect(() => {
    reset()
    setNatural(null)
  }, [eventId, image?.original_url, reset])

  // ★只保留 ESC★：方向键切图已随三图一起取消
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // ★滚轮缩放必须是非 passive 监听★：React 的 onWheel 默认 passive，
  // 里面调 preventDefault 不生效 → 背后页面会跟着一起滚。
  useEffect(() => {
    const el = stageRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      setZoom((z) => clampZoom(z * (e.deltaY < 0 ? 1.15 : 1 / 1.15)))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [clampZoom])

  if (typeof document === 'undefined') return null

  const base = apiUrl || ''
  const current = image || null
  const status = current?.status || (current?.original_url ? 'ready' : 'pending')
  /**
   * ★只有 `ready` 且有原图 URL 才能显示大图★ —— 其余情况一律走占位，
   * **绝不用 thumbnail 顶上**（放大后的缩略图 = 假证据）。
   */
  const ready = (status === 'ready' || !!current?.source_original_fallback)
    && !!current?.original_url
  const originalUrl = ready ? withVersion(base, current!.original_url, current?.version) : ''
  // 元信息缺失时用图片自身的解码尺寸兜底（naturalWidth 就是**原图**尺寸）
  const width = current?.width || natural?.w || 0
  const height = current?.height || natural?.h || 0

  return createPortal(
    <div
      data-alert-modal="1"
      data-alert-modal-event={eventId || ''}
      className="fixed inset-0 z-[60] flex flex-col bg-black/90 backdrop-blur-sm animate-card-enter"
      onClick={onClose}
    >
      {/* ── 顶栏（无 1/3 计数、无上一张/下一张） ────────────────────────── */}
      <div
        className="flex shrink-0 items-center gap-3 border-b border-white/10 px-4 py-2.5 text-white/90"
        onClick={(e) => e.stopPropagation()}
      >
        <span className="min-w-0 truncate text-xs">{title || '告警证据'}</span>
        <div className="ml-auto flex shrink-0 items-center gap-1">
          <button
            className="rounded p-1 hover:bg-white/10"
            onClick={() => setZoom((z) => clampZoom(z / 1.25))}
            title="缩小"
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </button>
          <span className="w-10 text-right font-mono text-[10px] text-white/60">
            {Math.round(zoom * 100)}%
          </span>
          <button
            className="rounded p-1 hover:bg-white/10"
            onClick={() => setZoom((z) => clampZoom(z * 1.25))}
            title="放大"
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </button>
          <button className="rounded p-1 hover:bg-white/10" onClick={reset} title="复位（双击大图同效）">
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
          <button className="ml-1 rounded p-1 hover:bg-white/10" onClick={onClose} title="关闭（Esc）">
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* ── 大图舞台：加载的一定是原图，没 ready 就明确说「处理中」 ──────── */}
      <div
        ref={stageRef}
        data-alert-modal-stage="1"
        data-alert-modal-status={status}
        className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden p-4"
        onClick={(e) => e.stopPropagation()}
      >
        {ready ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            data-alert-modal-image="1"
            data-alert-modal-src={current!.original_url}
            data-alert-modal-natural=""
            // ★key 必须含 eventId + url + version★：换证据时强制重新请求，
            // 否则浏览器可能复用上一张的解码结果（看的人以为换了图，其实没有）
            key={`${eventId || ''}-${current!.original_url}-${current?.version || ''}`}
            src={originalUrl}
            alt="告警证据"
            draggable={false}
            onLoad={(e) => {
              const el = e.currentTarget
              setNatural({ w: el.naturalWidth, h: el.naturalHeight })
              el.setAttribute('data-alert-modal-natural', `${el.naturalWidth}x${el.naturalHeight}`)
            }}
            onPointerDown={(e) => {
              if (e.button !== 0) return
              dragRef.current = { x: e.clientX, y: e.clientY, ox: offset.x, oy: offset.y }
              try { e.currentTarget.setPointerCapture(e.pointerId) } catch { /* 忽略 */ }
            }}
            onPointerMove={(e) => {
              const d = dragRef.current
              if (!d) return
              setOffset({ x: d.ox + (e.clientX - d.x), y: d.oy + (e.clientY - d.y) })
            }}
            onPointerUp={(e) => {
              dragRef.current = null
              try { e.currentTarget.releasePointerCapture(e.pointerId) } catch { /* 忽略 */ }
            }}
            onPointerCancel={() => { dragRef.current = null }}
            onDoubleClick={reset}
            style={{
              // ★80vh / 90vw + contain★：只改变**显示尺寸**，不动分辨率
              maxHeight: '80vh',
              maxWidth: '90vw',
              objectFit: 'contain',
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
              imageRendering: zoom >= PIXELATED_AT ? 'pixelated' : 'auto',
              cursor: zoom > 1 ? 'grab' : 'default',
            }}
            className="select-none"
          />
        ) : status === 'error' ? (
          <div className="flex flex-col items-center gap-2 text-white/70" data-alert-modal-placeholder="error">
            <X className="h-6 w-6 text-red-400/80" />
            <p className="text-xs">证据图生成失败</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 text-white/70" data-alert-modal-placeholder="pending">
            <Loader2 className="h-6 w-6 animate-spin" />
            <p className="text-xs">证据图处理中…</p>
            <p className="text-[10px] text-white/40">后台正在生成原图，稍后自动显示</p>
          </div>
        )}
      </div>

      {/* ── 图片信息：确认"我看的确实是原图" ───────────────────────────── */}
      <div
        className="flex shrink-0 flex-wrap items-center justify-center gap-2 px-4 pb-4 pt-1 text-[10px] text-white/50"
        onClick={(e) => e.stopPropagation()}
      >
        {ready ? (
          <>
            <span className="rounded border border-white/15 px-1.5 py-0.5 text-white/70">
              {current?.source_original_fallback ? '未标注原始帧' : String(current?.format || '').toUpperCase() || '原图'}
            </span>
            <span data-alert-modal-size="1" className="font-mono">
              {width && height ? `${width} × ${height}` : '尺寸读取中'}
            </span>
            {current?.bytes ? (
              <span className="font-mono">{humanKB(current.bytes)}</span>
            ) : null}
            {current?.bbox && current.bbox.length === 4 ? (
              <span className="font-mono text-white/35">
                bbox {current.bbox.map((v) => Math.round(v)).join(', ')}
              </span>
            ) : null}
            {current?.quality_ratio ? (
              <span className="font-mono text-white/35" title="证据图 / 源帧清晰度比值，≈1 表示处理链无损">
                ratio {current.quality_ratio.toFixed(3)}
              </span>
            ) : null}
            <a
              href={originalUrl}
              target="_blank"
              rel="noreferrer"
              className="ml-1 underline underline-offset-2 hover:text-white/80"
            >
              新标签页打开原图
            </a>
            {current?.source_url ? (
              <a
                href={withVersion(base, current.source_url, current?.version)}
                target="_blank"
                rel="noreferrer"
                className="underline underline-offset-2 hover:text-white/80"
                title="未画框的原始帧，用于人工比对框位置"
              >
                未标注原始帧
              </a>
            ) : null}
            {current?.source_original_fallback ? (
              <span className="text-amber-300/80">标注证据生成失败，当前显示原始帧</span>
            ) : null}
          </>
        ) : (
          <span className="text-white/40">
            {status === 'error' ? '该条告警的证据图未生成' : '原图生成后将显示实际尺寸与文件大小'}
          </span>
        )}
      </div>
    </div>,
    document.body,
  )
}
