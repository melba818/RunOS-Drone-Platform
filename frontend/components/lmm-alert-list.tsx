'use client'

import { useState } from 'react'
import { BellRing, Loader2 } from 'lucide-react'
import { toast } from 'sonner'

import { LmmAlertModal, type LmmEvidenceImage } from '@/components/lmm-alert-modal'

/**
 * ★告警记录里的一条★ = 一次 VLX 有效检出（1-Hit 即时告警）。
 *
 * 字段刻意只保留 **VLX 自己的东西**：时间、类别、描述、证据图。
 * 没有 `track_id` / `hit_frames` / `matched` 这些跟踪时代的字段 —— 系统里已经
 * 没有任何跟踪器，留着只会让人以为框还在被跟踪。
 */
export interface LmmEvent {
  id: string
  task_id: string
  /** 秒级时间戳（增量拉取用）；展示用 `time_str` */
  ts: number
  /** `HH:MM:SS` */
  time_str: string
  /** VLX 原始类别标签（person / swimmer …），按约定保留原文不汉化 */
  label: string
  label_zh: string
  /** 目标描述，形如「疑似人员」 */
  display_name: string
  /** 证据图**原图** URL（后端路径）；还没 ready 时是空串。★恒为 0 或 1 个元素★ */
  images?: string[]
  /**
   * ★证据图元信息★（后端 `image_meta`）：
   * `{index, status, frame_id, pts_ms, source_url, original_url,
   *   thumbnail_url, bbox, width, height, bytes, source_sharpness,
   *   sharpness, quality_ratio, version}`。
   *
   * ★1-Hit + 单证据图★：一条告警只有**一张**证据图（触发告警的那一次检出帧），
   * 所以这里取 `[0]` 就够 —— 不存在第二、第三张。
   *
   * 卡片显示**缩略图**，弹窗加载**原图** —— 两者必须分开：
   * 一张 1920×1080 原图在卡片上只占一百多像素宽（浪费），而缩略图当大图放大后
   * 全是马赛克，远距离目标根本认不出来。
   */
  image_meta?: LmmEvidenceImage[]
  /** 告警级证据状态：`pending` / `ready` / `error` */
  evidence_status?: string
  /**
   * ★主证据结构（单对象，不是数组）★
   * `{status, source_url, original_url, thumbnail_url, width, height, bytes,
   *   format, frame_id, pts_ms, bbox}`。
   * 1-Hit 之后一条告警恒为一张证据图，所以它是对象而非 `evidence[0..2]`。
   * 还没生成时三个 url 全是 `null`。
   */
  evidence?: {
    status?: string
    source_url?: string | null
    original_url?: string | null
    thumbnail_url?: string | null
    width?: number | null
    height?: number | null
    bytes?: number | null
    format?: string | null
    frame_id?: number | null
    bbox?: number[] | null
    source_original_fallback?: boolean
  } | null
  /**
   * 证据帧号。后端仍下发，**界面不展示** —— 帧号是给开发看的调试信息，
   * 对用户来说「哪一帧」没有意义，有意义的是"那时画面上确实有它"。
   */
  frames?: number[]
  /** VLX 给的像素框（xyxy）。**前端不画**：框已经由后端烧进证据图里了 */
  bbox?: number[]
  /** 语义状态（契约字段）。界面不再渲染徽章 —— 那是「算法状态」 */
  vlm_state?: 'pending' | 'confirmed' | 'rejected' | 'error'
}

interface Props {
  events: LmmEvent[]
  onClear?: () => void
  /** 后端地址。证据图是**后端路径**（`/api/lmm/...`），必须拼上它才能访问。 */
  apiUrl?: string
  canRecoverLocalVideo?: boolean
  onRecoverLocalFrame?: (event: LmmEvent) => Promise<LmmEvent>
}

function evidenceForEvent(e: LmmEvent): LmmEvidenceImage {
  const ev = e.evidence || null
  return ev
    ? {
        index: 0,
        status: ev.status || (ev.original_url ? 'ready' : 'pending'),
        original_url: ev.original_url || '',
        thumbnail_url: ev.thumbnail_url || '',
        source_url: ev.source_url || '',
        width: ev.width ?? undefined,
        height: ev.height ?? undefined,
        bytes: ev.bytes ?? undefined,
        format: ev.format ?? undefined,
        frame_id: ev.frame_id ?? undefined,
        bbox: ev.bbox || undefined,
        source_original_fallback: !!ev.source_original_fallback,
      }
    : e.image_meta && e.image_meta.length
      ? e.image_meta[0]
      : { index: 0, original_url: (e.images || [])[0] || '', status: (e.images || [])[0] ? 'ready' : 'pending' }
}

export function LmmAlertList({ events, onClear, apiUrl, canRecoverLocalVideo,
  onRecoverLocalFrame }: Props) {
  const [viewer, setViewer] = useState<
    { image: LmmEvidenceImage; title: string; eventId: string } | null
  >(null)

  if (events.length === 0) {
    return (
      <div className="flex h-full min-h-[200px] flex-col items-center justify-center gap-2 px-4 text-center">
        <BellRing className="h-6 w-6 text-muted-foreground/50" />
        <p className="text-xs text-muted-foreground">暂无告警</p>
        <p className="text-[10px] text-muted-foreground/70">
          发现目标后会在这里留下记录，点图片可放大查看
        </p>
      </div>
    )
  }

  const viewerEvent = viewer ? events.find((e) => e.id === viewer.eventId) : undefined
  const viewerImage = viewerEvent ? evidenceForEvent(viewerEvent) : viewer?.image

  return (
    <>
      <div className="space-y-2">
        {onClear && (
          <div className="flex items-center justify-between pb-1">
            <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              共 {events.length} 条
            </span>
            <button
              className="text-[10px] text-muted-foreground hover:text-foreground"
              onClick={onClear}
            >
              清空
            </button>
          </div>
        )}

        {/* 新的在上：告警列表最重要的是"最新发生了什么" */}
        {[...events].reverse().map((e) => {
          /**
           * ★卡片吃什么、弹窗吃什么★：优先读单对象 `e.evidence`（新主结构），
           * 没有再退回 `image_meta[0]`，都没有才退化成 `images[0]` 原图
           * 既当缩略图又当大图 —— 只多花流量，绝不会出现"弹窗里看压缩图"。
           *
           * ★只取第一张★：1-Hit 之后一条告警恒为一张证据图，
           * 这里不存在 `items.map(...)` 的三图网格了。
           */
          const image = evidenceForEvent(e)
          const status = image.status || (image.original_url ? 'ready' : 'pending')
          const ready = status === 'ready' && !!image.original_url
          const sourceFallback = status === 'error' && !!image.source_url
          const canRecover = status === 'error' && !!canRecoverLocalVideo && !!onRecoverLocalFrame
          const canOpen = ready || sourceFallback || canRecover
          const cardUrl = ready ? image.original_url : sourceFallback ? image.source_url : ''
          const title = e.display_name || e.label_zh || e.label || '疑似目标'
          return (
            <div
              key={e.id}
              data-alert-card={e.id}
              data-evidence-status={e.evidence_status || ''}
              className="rounded-lg border border-border bg-card p-2 transition-colors hover:bg-accent/40"
            >
              {/* 目标描述 + 告警时间 */}
              <div className="flex items-baseline gap-2">
                <span
                  className="min-w-0 flex-1 truncate text-xs font-medium text-foreground"
                  title={title}
                >
                  {title}
                </span>
                <span className="shrink-0 font-mono text-[10px] text-muted-foreground">
                  {e.time_str}
                </span>
              </div>

              {/*
                这里原本还有一行「目标类别」（VLX 原始 label，如「人」）。
                已删除：标题已经把目标的说法讲清楚了（「疑似人员」），
                再挂一行裸标签只是把同一件事说两遍，还容易被当成两个不同的东西。
              */}

              {/*
                ★唯一一张证据图★：占卡片主要宽度、16:9。
                以前是三张 96 px 小图（三帧确认留下的），现在一条告警只对应
                触发它的那一帧 —— 单张放大了，用户在右侧列表就能初步看清目标。
                列表始终走服务端缩略图接口；打开弹窗才请求原图，避免加载高清 PNG。
              */}
              <button
                data-evidence-thumb={`${e.id}-0`}
                data-evidence-status={status}
                type="button"
                disabled={!canOpen}
                onClick={async () => {
                  if (ready) {
                    setViewer({ image, title, eventId: e.id })
                    return
                  }
                  if (sourceFallback) {
                    setViewer({
                      image: { ...image, original_url: image.source_url || '',
                        source_original_fallback: true },
                      title, eventId: e.id,
                    })
                    return
                  }
                  if (!canRecover || !onRecoverLocalFrame) return
                  try {
                    const updated = await onRecoverLocalFrame(e)
                    setViewer({ image: evidenceForEvent(updated), title, eventId: e.id })
                  } catch (err) {
                    toast.error(err instanceof Error ? err.message : '恢复原始帧失败')
                  }
                }}
                className="mt-1.5 block w-full overflow-hidden rounded border border-border bg-black/80 disabled:cursor-default disabled:opacity-60"
                title={
                  ready
                    ? '点击查看原图'
                    : sourceFallback
                      ? '点击查看未处理的原始帧'
                      : canRecover
                        ? '点击从本地视频恢复原始帧并放大'
                        : status === 'error'
                          ? '证据图生成失败'
                      : '证据图处理中…'
                }
              >
                {cardUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={`${apiUrl || ''}${cardUrl}${cardUrl.includes('?') ? '&' : '?'}thumbnail=1`}
                    alt="告警证据"
                    loading="lazy"
                    decoding="async"
                    className="aspect-video w-full object-cover"
                  />
                ) : status === 'error' ? (
                  <span className="flex aspect-video w-full flex-col items-center justify-center gap-1 text-[11px] text-red-500/80">
                    <span>生成失败</span>
                    {canRecover && <span className="text-[10px] text-muted-foreground">点此恢复原帧并放大</span>}
                  </span>
                ) : (
                  <span className="flex aspect-video w-full flex-col items-center justify-center gap-1 text-[11px] text-muted-foreground/70">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    证据图处理中…
                  </span>
                )}
              </button>
            </div>
          )
        })}
      </div>

      {viewer && (
        <LmmAlertModal
          image={viewerImage || viewer.image}
          eventId={viewer.eventId}
          title={viewer.title}
          apiUrl={apiUrl}
          onClose={() => setViewer(null)}
        />
      )}
    </>
  )
}
