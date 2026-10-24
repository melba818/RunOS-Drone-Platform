import { NextRequest, NextResponse } from 'next/server'
import { resolveTenantBackend } from '@/lib/tenant-backend'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

async function proxy(request: NextRequest, context: { params: Promise<{ segments: string[] }> }) {
  const { segments } = await context.params
  if (!segments.length || segments.some(part => part === '.' || part === '..')) {
    return NextResponse.json({ detail: 'Invalid backend path' }, { status: 400 })
  }

  const endpoint = `/${segments.join('/')}`
  if (!endpoint.startsWith('/api/')) {
    return NextResponse.json({ detail: 'Only RunOS API routes can be proxied' }, { status: 404 })
  }

  const token = request.cookies.get('runos_session')?.value
  if (!token) return NextResponse.json({ detail: 'RunHive sign-in required' }, { status: 401 })

  let backendUrl: string
  try {
    backendUrl = (await resolveTenantBackend(token)).url
  } catch {
    return NextResponse.json({ detail: 'RunOS workspace is unavailable' }, { status: 503 })
  }

  const headers = new Headers()
  const contentType = request.headers.get('content-type')
  if (contentType) headers.set('content-type', contentType)
  const accept = request.headers.get('accept')
  if (accept) headers.set('accept', accept)
  const range = request.headers.get('range')
  if (range) headers.set('range', range)
  headers.set('authorization', `Bearer ${token}`)

  const method = request.method.toUpperCase()
  const init: RequestInit & { duplex?: 'half' } = {
    method,
    headers,
    cache: 'no-store',
  }
  if (method !== 'GET' && method !== 'HEAD') {
    init.body = request.body
    init.duplex = 'half'
  }

  try {
    const upstream = await fetch(`${backendUrl}${endpoint}${request.nextUrl.search}`, init)
    if (upstream.status >= 500) {
      void upstream.body?.cancel().catch(() => {})
      return NextResponse.json({ detail: 'RunOS backend request failed' }, { status: upstream.status })
    }
    const responseHeaders = new Headers()
    for (const name of ['content-type', 'content-disposition', 'content-length', 'content-range', 'accept-ranges']) {
      const value = upstream.headers.get(name)
      if (value) responseHeaders.set(name, value)
    }
    return new NextResponse(upstream.body, {
      status: upstream.status,
      headers: responseHeaders,
    })
  } catch {
    return NextResponse.json({ detail: 'RunOS backend is unavailable' }, { status: 503 })
  }
}

export const GET = proxy
export const HEAD = proxy
export const POST = proxy
export const PUT = proxy
export const PATCH = proxy
export const DELETE = proxy
