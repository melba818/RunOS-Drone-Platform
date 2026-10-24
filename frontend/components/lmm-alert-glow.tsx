'use client'

/**
 * ★VLX 告警的全部视觉表达★：整个视频窗口外围的红色呼吸闪烁。
 *
 * 刻意**不画检测框**：告警说的是"这一路视频里出现目标了"，不是"第 N 帧的某个
 * 像素矩形里有东西"。把框画出来会把"有事发生"变成"跟踪还在跑"，而且框会遮挡
 * 目标本身。多个目标同时命中时也只整体闪一次，不按位置逐个闪（那会变成一片噪声）。
 *
 * 纯 CSS 动画（不是 Canvas / rAF）：rAF 在窗口最小化或被完全遮挡时**完全停摆**，
 * 光晕会静默消失；CSS 动画由合成器驱动，不受 `visibilityState` 影响。
 *
 * 用法：放进一个 `relative` 的容器里，容器**不要** `overflow-hidden`（会裁掉外圈）。
 */
export function LmmAlertGlow({ active }: { active: boolean }) {
  if (!active) return null
  return (
    <div
      data-alert-glow="1"
      aria-hidden
      className="alert-glow-ring alert-glow-tint pointer-events-none absolute -inset-1 z-20 rounded-[10px]"
    />
  )
}
