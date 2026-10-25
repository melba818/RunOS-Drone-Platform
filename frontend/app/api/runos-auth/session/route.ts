import { NextRequest, NextResponse } from 'next/server'
import { ensureTenantBackend } from '@/lib/tenant-backend'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_URL = (process.env.BACKEND_URL || 'http://127.0.0.1:8010').replace(/\/$/, '')

export async function GET(request: NextRequest) {
  const token = request.cookies.get('runos_session')?.value
  if (!token) return NextResponse.json({ detail: 'RunOS sign-in required' }, { status: 401 })

  try {
    const upstream = await fetch(`${BACKEND_URL}/api/auth/me`, {
      headers: { authorization: `Bearer ${token}` },
      cache: 'no-store',
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
    const tenant = await ensureTenantBackend(Number(body.admin_id))
    return NextResponse.json(
      { ...body, backend_port: tenant.port },
      { headers: { 'Cache-Control': 'no-store' } },
    )
  } catch {
    return NextResponse.json({ detail: 'RunOS backend is unavailable' }, { status: 503 })
  }
}
