import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_URL = (process.env.BACKEND_URL || 'http://127.0.0.1:8010').replace(/\/$/, '')

export async function POST(request: NextRequest) {
  let ticket = ''
  try {
    ticket = String((await request.json())?.ticket || '')
  } catch {
    return NextResponse.json({ detail: 'Invalid sign-in ticket' }, { status: 400 })
  }
  if (!/^[a-f0-9]{64}$/.test(ticket)) {
    return NextResponse.json({ detail: 'Invalid sign-in ticket' }, { status: 400 })
  }

  try {
    const upstream = await fetch(`${BACKEND_URL}/api/auth/exchange`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ ticket }),
      cache: 'no-store',
      signal: AbortSignal.timeout(10_000),
    })
    const result = await upstream.json()
    if (!upstream.ok) {
      return NextResponse.json({ detail: upstream.status >= 500 ? 'RunHive sign-in service is unavailable' : (result.detail || 'RunHive sign-in failed') }, { status: upstream.status })
    }

    if (typeof result.access_token !== 'string' || !result.access_token) {
      return NextResponse.json({ detail: 'Invalid RunHive sign-in response' }, { status: 502 })
    }

    const response = NextResponse.json({ ok: true, account: result.account })
    response.cookies.set('runos_session', result.access_token, {
      httpOnly: true,
      secure: request.headers.get('x-forwarded-proto')?.split(',')[0].trim() === 'https' || request.nextUrl.protocol === 'https:',
      sameSite: 'lax',
      path: '/',
      maxAge: Number(result.expires_in) || 28800,
    })
    return response
  } catch (error) {
    if (error instanceof Error && error.name === 'TimeoutError') {
      return NextResponse.json({ detail: 'RunOS 登录服务响应超时，请重新连接。' }, { status: 503 })
    }
    return NextResponse.json({ detail: 'RunOS backend is unavailable' }, { status: 503 })
  }
}
