import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const CONTROL_URL = (process.env.BACKEND_URL || 'http://127.0.0.1:8010').replace(/\/$/, '')
const TENANT_PORT_BASE = 20000

type TenantStartRegistry = typeof globalThis & {
  __runosTenantStarts?: Map<number, Promise<void>>
}

function startRegistry(): Map<number, Promise<void>> {
  const host = globalThis as TenantStartRegistry
  host.__runosTenantStarts ??= new Map<number, Promise<void>>()
  return host.__runosTenantStarts
}

export function tenantBackendPort(adminId: number): number {
  if (!Number.isSafeInteger(adminId) || adminId <= 0 || adminId > 45535) {
    throw new Error('RunHive account ID is outside the local RunOS worker port range')
  }
  return adminId === 1 ? 8010 : TENANT_PORT_BASE + adminId
}

async function workerHealth(port: number): Promise<number | null> {
  try {
    const response = await fetch(`http://127.0.0.1:${port}/api/health`, {
      cache: 'no-store',
      signal: AbortSignal.timeout(1500),
    })
    if (!response.ok) return null
    const body = await response.json() as { status?: string; tenant_admin_id?: number }
    return body.status === 'healthy' && Number.isSafeInteger(body.tenant_admin_id)
      ? Number(body.tenant_admin_id)
      : null
  } catch {
    return null
  }
}

export async function authenticatedAdminId(token: string): Promise<number> {
  const response = await fetch(`${CONTROL_URL}/api/auth/me`, {
    headers: { authorization: `Bearer ${token}` },
    cache: 'no-store',
  })
  const account = await response.json() as { admin_id?: number }
  const adminId = Number(account.admin_id)
  if (!response.ok || !Number.isSafeInteger(adminId) || adminId <= 0) {
    throw new Error('RunOS session is invalid or expired')
  }
  return adminId
}

export async function ensureTenantBackend(adminId: number): Promise<{ adminId: number; port: number; url: string }> {
  // Standalone deployment (containers): the control backend serves all
  // tenants, and worker ports live in another container, so localhost
  // worker probing and the Windows worker spawner do not apply.
  if (process.env.RUNOS_STANDALONE === "1") {
    return { adminId, port: tenantBackendPort(adminId), url: CONTROL_URL }
  }
  const port = tenantBackendPort(adminId)
  const url = `http://127.0.0.1:${port}`
  const currentOwner = await workerHealth(port)
  if (currentOwner !== null) {
    if (currentOwner !== adminId) throw new Error(`RunOS worker port ${port} belongs to another account`)
    return { adminId, port, url }
  }

  if (process.platform !== 'win32' && adminId !== 1) {
    throw new Error('Automatic RunOS workspace workers are configured for the local Windows deployment')
  }

  const registry = startRegistry()
  let pending = registry.get(adminId)
  if (!pending) {
    pending = (async () => {
      const projectRoot = process.env.RUNOS_PROJECT_ROOT || process.cwd()
      const script = `${projectRoot.replace(/[\\/]$/, '')}\\start-runos-tenant.ps1`
      await execFileAsync('powershell.exe', [
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script, '-AdminId', String(adminId),
      ], { windowsHide: true, timeout: 180000, maxBuffer: 256 * 1024 })
    })()
    registry.set(adminId, pending)
  }
  try {
    await pending
  } finally {
    registry.delete(adminId)
  }

  const startedOwner = await workerHealth(port)
  if (startedOwner !== adminId) throw new Error('The isolated RunOS workspace service did not become healthy')
  return { adminId, port, url }
}

export async function resolveTenantBackend(token: string) {
  const adminId = await authenticatedAdminId(token)
  return ensureTenantBackend(adminId)
}
