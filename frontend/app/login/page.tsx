'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

/**
 * Standalone sign-in page for the local admin account.
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
        background: '#f5f8fc',
        color: '#0f172a',
        fontFamily: 'system-ui, sans-serif',
      }}
    >
      <form
        onSubmit={onSubmit}
        style={{
          width: 380,
          padding: 32,
          borderRadius: 16,
          background: '#ffffff',
          border: '1px solid #e2e8f0',
          boxShadow: '0 1px 3px rgba(15, 23, 42, 0.08)',
          display: 'flex',
          flexDirection: 'column',
          gap: 14,
        }}
      >
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 600 }}>RunOS</h1>
        <p style={{ margin: 0, color: '#64748b', fontSize: 14 }}>
          Standalone sign-in
        </p>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 14, color: '#334155' }}>
          Username
          <input
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            autoComplete="username"
            style={{ padding: '10px 12px', borderRadius: 8, border: '1px solid #cbd5e1', background: '#ffffff', color: '#0f172a' }}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 14, color: '#334155' }}>
          Password
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
            style={{ padding: '10px 12px', borderRadius: 8, border: '1px solid #cbd5e1', background: '#ffffff', color: '#0f172a' }}
          />
        </label>
        {error ? <p style={{ margin: 0, color: '#dc2626', fontSize: 13 }}>{error}</p> : null}
        <button
          type="submit"
          disabled={busy}
          style={{ padding: '11px 12px', borderRadius: 8, border: 0, background: '#2f6feb', color: '#fff', fontSize: 15, cursor: 'pointer' }}
        >
          {busy ? 'Signing in…' : 'Sign in'}
        </button>
        <p style={{ margin: 0, color: '#64748b', fontSize: 12 }}>
          Default demo account: admin / runos-demo. Change the password
          in .env before exposing this deployment.
        </p>
      </form>
    </main>
  )
}
