// This API route acts as a proxy to the Python backend
// In production, the Python backend runs separately on port 8000

import { NextRequest, NextResponse } from "next/server"
import { resolveTenantBackend } from "@/lib/tenant-backend"

// Preserve the upstream status; never expose internal 5xx response bodies.
function forwardBackendResponse(response: Response) {
  if (response.status >= 500) {
    void response.body?.cancel().catch(() => {})
    return NextResponse.json({ error: "Backend request failed." }, { status: response.status })
  }
  const headers = new Headers()
  const contentType = response.headers.get("content-type")
  if (contentType) headers.set("content-type", contentType)
  return new NextResponse(response.body, { status: response.status, headers })
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const endpoint = searchParams.get("endpoint") || ""
  const token = request.cookies.get("runos_session")?.value
  if (!token || !endpoint.startsWith("/api/")) {
    return NextResponse.json({ detail: "RunOS sign-in required" }, { status: 401 })
  }
  
  try {
    const backend = await resolveTenantBackend(token)
    const response = await fetch(`${backend.url}${endpoint}`, {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      cache: "no-store",
    })
    
    return forwardBackendResponse(response)
  } catch {
    return NextResponse.json(
      { error: "Backend unavailable. Please ensure the Python backend is running." },
      { status: 503 }
    )
  }
}

export async function POST(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const endpoint = searchParams.get("endpoint") || ""
  const token = request.cookies.get("runos_session")?.value
  if (!token || !endpoint.startsWith("/api/")) {
    return NextResponse.json({ detail: "RunOS sign-in required" }, { status: 401 })
  }
  
  try {
    const body = await request.json()
    const backend = await resolveTenantBackend(token)
    
    const response = await fetch(`${backend.url}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(body),
      cache: "no-store",
    })
    
    return forwardBackendResponse(response)
  } catch {
    return NextResponse.json(
      { error: "Backend unavailable. Please ensure the Python backend is running." },
      { status: 503 }
    )
  }
}
