"use client"

import { useEffect, useRef } from "react"

interface VisosLogoProps {
  size?: number
  showText?: boolean
}

// Clean circular geometry and hand-tuned Bezier contours for crisp rendering at sidebar size.
const MARK_PATH = "M244 73 A164 164 0 1 1 244 401 A164 164 0 1 1 244 73 Z M244 81 A156 156 0 1 0 244 393 A156 156 0 1 0 244 81 Z M296 149 C274 128 249 116 222 118 C156 122 110 180 109 246 C108 317 162 378 234 378 C251 378 267 375 282 369 C266 372 250 373 234 372 C167 372 117 315 117 246 C117 184 160 130 222 126 C248 124 271 132 289 147 Z M297 146 C269 137 242 138 219 150 C185 167 163 199 162 237 C160 294 204 341 262 343 C298 345 329 331 352 306 C324 327 296 338 263 336 C211 333 169 293 170 237 C171 204 191 174 222 158 C245 146 271 145 296 154 C309 159 322 164 334 169 C346 174 350 185 360 195 C366 202 375 209 378 216 C381 222 377 228 371 233 C366 237 360 237 352 234 C339 229 327 225 313 222 C298 219 288 213 277 203 C265 192 253 185 238 191 C222 197 213 212 212 230 C210 261 237 287 271 295 C305 303 338 292 361 268 C333 287 304 294 274 288 C244 282 219 260 220 231 C220 215 228 202 241 198 C254 194 263 201 273 210 C285 221 297 227 312 230 C326 233 338 237 350 242 C360 246 369 245 376 239 C385 232 390 222 385 212 C381 204 372 197 365 190 C356 181 351 169 337 162 C325 156 311 151 297 146 Z M375 252 C394 288 381 328 352 354 C326 378 291 390 261 393 C299 393 334 380 360 356 C390 328 400 287 375 252 Z"
const EYE_PATH = "M316 180 C329 180 339 188 343 202 C329 203 319 195 316 180 Z"
// Muzzle tip (380, 224) relative to the square SVG viewport center (244, 237).
const MOUTH_BEARING = -5.46021259

// Keep the existing component API for the sidebar; branding is now RunOS.
export function VisosLogo({ size = 200, showText = true }: VisosLogoProps) {
  const iconRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    let frame = 0
    let rotation = 0
    let pointer = { x: 0, y: 0 }
    const update = () => {
      frame = 0
      const icon = iconRef.current
      if (!icon) return
      const rect = icon.getBoundingClientRect()
      const dx = pointer.x - (rect.left + rect.width / 2)
      const dy = pointer.y - (rect.top + rect.height / 2)
      if (Math.hypot(dx, dy) < 1) return
      const target = Math.atan2(dy, dx) * 180 / Math.PI - MOUTH_BEARING
      let delta = target - rotation
      while (delta > 180) delta -= 360
      while (delta < -180) delta += 360
      rotation += delta
      icon.style.transform = `rotate(${rotation}deg)`
    }
    const followPointer = (event: MouseEvent) => {
      pointer = { x: event.clientX, y: event.clientY }
      if (!frame) frame = window.requestAnimationFrame(update)
    }
    window.addEventListener("mousemove", followPointer, { passive: true })
    return () => {
      window.removeEventListener("mousemove", followPointer)
      if (frame) window.cancelAnimationFrame(frame)
    }
  }, [])

  return (
    <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: size * 0.2 }}>
      <svg
        ref={iconRef}
        width={size}
        height={size}
        viewBox="62 55 364 364"
        shapeRendering="geometricPrecision"
        style={{ display: "block", transformOrigin: "center center", transition: "transform 0.15s ease-out" }}
        role="img"
        aria-label="RunOS logo"
        xmlns="http://www.w3.org/2000/svg"
      >
        <title>RunOS</title>
        {MARK_PATH.split(/(?=M)/).filter(Boolean).reduce<string[]>((paths, part, index) => {
          if (index === 1) paths[0] += part
          else paths.push(part)
          return paths
        }, []).map((path, index) => <path key={index} d={path} fill="var(--logo-ink)" fillRule="evenodd" />)}
        <path d={EYE_PATH} fill="var(--logo-accent)" />
      </svg>
      {showText && (
        <div style={{ width: size * 3.85, height: size, position: "relative", flex: "0 0 auto" }}>
          <div style={{
            display: "flex",
            alignItems: "center",
            position: "absolute",
            left: 0,
            top: "50%",
            transform: "translateY(-50%) scaleX(0.55)",
            transformOrigin: "left center",
            fontFamily: "'Arial Black', 'Avenir Next', system-ui, sans-serif",
            fontSize: size * 1.28,
            fontWeight: 800,
            letterSpacing: "-0.025em",
            lineHeight: 1,
            whiteSpace: "nowrap",
          }}>
            <span style={{ color: "var(--logo-ink)" }}>Run</span>
            <span style={{ color: "var(--logo-accent)" }}>OS</span>
          </div>
        </div>
      )}
    </div>
  )
}
