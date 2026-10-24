import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_URL = (process.env.BACKEND_URL || 'http://127.0.0.1:8010').replace(/\/$/, '')

/**
 * Standalone login proxy. Only active when the backend runs in local
 * auth mode (RUNOS_AUTH_MODE=local). Forwards the credentials to the
 * backend and stores the returned session token in the same
 * `runos_session` cookie the ticket-exchange flow uses.
 */
export async function POST(request: NextRequest) {
  let body: { username?: unknown; password?: unknown }
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ detail: 'Invalid request' }, { status: 400 })
  }
  const username = String(body.username || '')
  const password = String(body.password || '')
  if (!username || !password) {
    return NextResponse.json({ detail: 'Username and password are required' }, { status: 400 })
  }

  try {
    const upstream = await fetch(`${BACKEND_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ username, password }),
      cache: 'no-store',
      signal: AbortSignal.timeout(10_000),
    })
    const result = await upstream.json()
    if (!upstream.ok) {
      return NextResponse.json(
        { detail: result.detail || 'Sign-in failed' },
        { status: upstream.status },
      )
    }
    if (typeof result.access_token !== 'string' || !result.access_token) {
      return NextResponse.json({ detail: 'Invalid sign-in response' }, { status: 502 })
    }
    const response = NextResponse.json({ ok: true, account: result.account })
    response.cookies.set('runos_session', result.access_token, {
      httpOnly: true,
      secure: request.headers.get('x-forwarded-proto')?.split(',')[0].trim() === 'https' || request.nextUrl.protocol === 'https:',
      sameSite: 'lax',
      path: '/',
      maxAge: 8 * 60 * 60,
    })
    return response
  } catch {
    return NextResponse.json({ detail: 'Backend unavailable' }, { status: 502 })
  }
}
