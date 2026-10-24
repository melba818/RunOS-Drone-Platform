'use client'

import { useEffect, useState, type ReactNode } from 'react'

type GateState = 'loading' | 'ready' | 'denied'
const isPreview = process.env.NEXT_PUBLIC_RUNHIVE_PREVIEW === '1'

function notifyRunHiveParent(type: 'runos-auth-ready' | 'runos-auth-error', message?: string) {
  if (isPreview || typeof window === 'undefined' || window.parent === window) return

  let parentOrigin = 'http://127.0.0.1:1819'
  try {
    parentOrigin = document.referrer
      ? new URL(document.referrer).origin
      : new URL(process.env.NEXT_PUBLIC_RUNHIVE_URL || parentOrigin).origin
  } catch {
    // Keep the local RunHive origin fallback.
  }

  window.parent.postMessage({ source: 'runos-auth-gate', type, message }, parentOrigin)
}

export function RunHiveAuthGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<GateState>('loading')
  const [message, setMessage] = useState('正在验证 RunHive 登录状态…')
  const isEmbedded = typeof window !== 'undefined' && (
    window.parent !== window || new URLSearchParams(window.location.search).get('rh_embed') === '1'
  )

  useEffect(() => {
    let active = true

    const markReadyAndWarmWorkspace = () => {
      if (active) setState('ready')
      void fetch('/api/runos-auth/session', {
        cache: 'no-store',
        signal: AbortSignal.timeout(180_000),
      }).catch(() => undefined)
    }

    async function authenticate() {
      try {
        const hash = window.location.hash.startsWith('#') ? window.location.hash.slice(1) : window.location.hash
        const hashParams = new URLSearchParams(hash)
        const ticket = hashParams.get('ticket')
        const expectedAdminId = Number(hashParams.get('admin_id'))
        if (ticket) {
          window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}`)

          // A previous RunHive login may already have left a valid RunOS
          // session in this browser. Reuse it only for the same account, so
          // opening the workspace does not wait for a fresh SSO exchange.
          if (Number.isSafeInteger(expectedAdminId) && expectedAdminId > 0) {
            try {
              const identity = await fetch('/api/runos-auth/identity', {
                cache: 'no-store',
                signal: AbortSignal.timeout(2500),
              })
              if (identity.ok) {
                const account = await identity.json() as { admin_id?: number }
                if (Number(account.admin_id) === expectedAdminId) {
                  markReadyAndWarmWorkspace()
                  return
                }
              }
            } catch {
              // A cache miss or slow identity check falls through to the
              // single-use ticket below.
            }
          }

          const exchange = await fetch('/api/runos-auth/exchange', {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ ticket }),
            cache: 'no-store',
            signal: AbortSignal.timeout(10_000),
          })
          const result = await exchange.json()
          if (!exchange.ok) throw new Error(isPreview ? '测试登录票据已失效，请重新运行 Open-RunHivePreview.ps1。' : result.detail || '登录票据已失效，请从应用中心重新打开 RunOS')

          // The ticket is redeemed once. Show the workspace immediately while
          // its account-specific API worker warms in the background.
          markReadyAndWarmWorkspace()
          return
        }

        const identity = await fetch('/api/runos-auth/identity', {
          cache: 'no-store',
          signal: AbortSignal.timeout(4000),
        })
        if (!identity.ok) throw new Error(isPreview ? '请先登录独立测试账户：在 Windows 运行 Open-RunHivePreview.ps1 获取短时登录链接。' : '请先登录 RunHive，再从“应用中心”打开 RunOS')
        markReadyAndWarmWorkspace()
      } catch (error) {
        if (!active) return
        setMessage(error instanceof Error ? error.message : 'RunHive 登录验证失败')
        setState('denied')
      }
    }

    void authenticate()
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (state === 'ready') notifyRunHiveParent('runos-auth-ready')
    if (state === 'denied') notifyRunHiveParent('runos-auth-error', message)
  }, [state, message])

  if (state === 'ready') return children
  if (state === 'loading') return isPreview ? <main className="flex min-h-screen items-center justify-center">正在连接独立测试预览…</main> : null
  if (isEmbedded) return null

  return (
    <main className="flex min-h-screen items-center justify-center bg-background px-6 text-foreground">
      <section className="max-w-lg rounded-2xl border border-border bg-card p-8 text-center shadow-sm">
        <h1 className="text-xl font-semibold">RunOS 账号验证</h1>
        <p className="mt-3 text-sm text-muted-foreground">{message}</p>
        {state === 'denied' && !isPreview && (
          <a
            className="mt-6 inline-flex rounded-lg bg-primary px-4 py-2 text-sm text-primary-foreground"
            href={process.env.NEXT_PUBLIC_RUNHIVE_URL || 'http://127.0.0.1:1819'}
            target="_top"
            rel="noreferrer"
          >
            返回 RunHive
          </a>
        )}
      </section>
    </main>
  )
}
