'use client'

import { useEffect, useState, type ReactNode } from 'react'
import { usePathname, useRouter } from 'next/navigation'

/**
 * Local-only auth gate for standalone RunOS.
 *
 * Checks the local admin session (the `runos_session` cookie set by
 * /login) via /api/runos-auth/identity. Authenticated users see the
 * workspace immediately; everyone else is sent to /login. The login
 * page itself bypasses the gate so a fresh browser can sign in.
 */
export function LocalAuthGate({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false)
  const pathname = usePathname()
  const router = useRouter()

  useEffect(() => {
    if (pathname === '/login') {
      setReady(true)
      return
    }
    let active = true

    async function authenticate() {
      try {
        const identity = await fetch('/api/runos-auth/identity', {
          cache: 'no-store',
          signal: AbortSignal.timeout(4000),
        })
        if (identity.ok) {
          if (active) setReady(true)
          void fetch('/api/runos-auth/session', {
            cache: 'no-store',
            signal: AbortSignal.timeout(180_000),
          }).catch(() => undefined)
          return
        }
      } catch {
        // Unreachable backend or no session: fall through to /login.
      }
      if (active) router.push('/login')
    }

    void authenticate()
    return () => {
      active = false
    }
  }, [pathname, router])

  if (ready) return children
  return null
}
