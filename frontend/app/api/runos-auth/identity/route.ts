import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_URL = (process.env.BACKEND_URL || 'http://127.0.0.1:8010').replace(/\/$/, '')

/** Validate the existing browser session without waiting for a workspace worker to start. */
export async function GET(request: NextRequest) {
  const token = request.cookies.get('runos_session')?.value
  if (!token) return NextResponse.json({ detail: 'RunHive sign-in required' }, { status: 401 })

  try {
    const upstream = await fetch(`${BACKEND_URL}/api/auth/me`, {
      headers: { authorization: `Bearer ${token}` },
      cache: 'no-store',
      signal: AbortSignal.timeout(2500),
    })
    const body = await upstream.json()
    if (!upstream.ok) {
      const response = NextResponse.json(
        upstream.status >= 500 ? { detail: 'RunOS authentication service is unavailable' } : body,
        { status: upstream.status, headers: { 'Cache-Control': 'no-store' } },
      )
      response.cookies.delete('runos_session')
      return response
    }
    return NextResponse.json(body, { headers: { 'Cache-Control': 'no-store' } })
  } catch (error) {
    const timedOut = error instanceof Error && error.name === 'TimeoutError'
    return NextResponse.json(
      { detail: timedOut ? 'RunOS 登录状态检查超时' : 'RunOS backend is unavailable' },
      { status: 503, headers: { 'Cache-Control': 'no-store' } },
    )
  }
}
