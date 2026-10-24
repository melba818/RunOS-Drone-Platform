'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

/**
 * Standalone sign-in page. Used only when the backend runs in local
 * auth mode; the portal ticket flow (RunHive mode) does not use it.
 */
export default function LoginPage() {
  const router = useRouter()
  const [username, setUsername] = useState('admin')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault()
    setError('')
    setBusy(true)
    try {
      const response = await fetch('/api/runos-auth/login', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      const result = await response.json().catch(() => ({}))
      if (!response.ok) {
        setError(result.detail || 'Sign-in failed')
        return
      }
      router.push('/')
      router.refresh()
    } catch {
      setError('Backend unavailable')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#0f1115',
        color: '#e6e6e6',
        fontFamily: 'system-ui, sans-serif',
      }}
    >
      <form
        onSubmit={onSubmit}
        style={{
          width: 360,
          padding: 28,
          borderRadius: 12,
          background: '#171a21',
          display: 'flex',
          flexDirection: 'column',
          gap: 14,
        }}
      >
        <h1 style={{ margin: 0, fontSize: 22 }}>RunOS</h1>
        <p style={{ margin: 0, color: '#9aa3b2', fontSize: 14 }}>
          Standalone sign-in
        </p>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 14 }}>
          Username
          <input
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            autoComplete="username"
            style={{ padding: '10px 12px', borderRadius: 8, border: '1px solid #2a2f3a', background: '#0f1115', color: '#e6e6e6' }}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 14 }}>
          Password
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
            style={{ padding: '10px 12px', borderRadius: 8, border: '1px solid #2a2f3a', background: '#0f1115', color: '#e6e6e6' }}
          />
        </label>
        {error ? <p style={{ margin: 0, color: '#ff7a7a', fontSize: 13 }}>{error}</p> : null}
        <button
          type="submit"
          disabled={busy}
          style={{ padding: '11px 12px', borderRadius: 8, border: 0, background: '#2f6feb', color: '#fff', fontSize: 15, cursor: 'pointer' }}
        >
          {busy ? 'Signing in…' : 'Sign in'}
        </button>
        <p style={{ margin: 0, color: '#9aa3b2', fontSize: 12 }}>
          Default demo account: admin / runos-demo. Change the password
          in .env before exposing this deployment.
        </p>
      </form>
    </main>
  )
}
