'use client'

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react'

export interface AppSettings {
  apiUrl: string
  modelsPath: string
  datasetsPath: string
  outputPath: string
  useGpu: boolean
  gpuDevice: string
  darkMode: boolean
  autoSave: boolean
  notifications: boolean
}

const DEFAULTS: AppSettings = {
  apiUrl: '/api/proxy',
  modelsPath: '/models',
  datasetsPath: '/datasets',
  outputPath: '/output',
  useGpu: true,
  gpuDevice: '0',
  darkMode: false,
  autoSave: true,
  notifications: true,
}

const STORAGE_KEY = 'cv-dataset-manager-settings'

interface SettingsContextValue {
  settings: AppSettings
  updateSettings: (patch: Partial<AppSettings>) => void
  saveSettings: (patch?: Partial<AppSettings>) => Promise<boolean>
  isSaving: boolean
  lastSaved: Date | null
}

const SettingsContext = createContext<SettingsContextValue | null>(null)

function loadFromStorage(): AppSettings {
  if (typeof window === 'undefined') return DEFAULTS
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULTS
    return { ...DEFAULTS, ...JSON.parse(raw), apiUrl: '/api/proxy', darkMode: false }
  } catch {
    return DEFAULTS
  }
}

function persistToStorage(s: AppSettings) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s))
  } catch {
    // storage quota or SSR — ignore
  }
}

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<AppSettings>(DEFAULTS)
  const [isSaving, setIsSaving] = useState(false)
  const [lastSaved, setLastSaved] = useState<Date | null>(null)
  // Hydrate from localStorage on mount
  useEffect(() => {
    const stored = loadFromStorage()
    setSettings(stored)
    persistToStorage(stored)

    void fetch('/api/proxy/api/settings', { cache: 'no-store' })
      .then(async response => response.ok ? response.json() : null)
      .then(server => {
        if (!server) return
        const next = {
          ...stored,
          apiUrl: '/api/proxy',
          modelsPath: server.models_path ?? stored.modelsPath,
          datasetsPath: server.datasets_path ?? stored.datasetsPath,
          outputPath: server.output_path ?? stored.outputPath,
          useGpu: server.use_gpu ?? stored.useGpu,
          gpuDevice: server.gpu_device ?? stored.gpuDevice,
        }
        setSettings(next)
        persistToStorage(next)
      })
      .catch(() => {})
  }, [])

  const updateSettings = useCallback((patch: Partial<AppSettings>) => {
    setSettings(prev => {
      const next = { ...prev, ...patch, darkMode: false }
      persistToStorage(next)
      return next
    })
  }, [])

  /**
   * Persist settings to localStorage AND push to the backend API.
   * Accepts an optional patch so the caller can do one combined update+save.
   */
  const saveSettings = useCallback(async (patch?: Partial<AppSettings>) => {
    setIsSaving(true)
    const next = { ...(patch ? { ...settings, ...patch } : settings), apiUrl: '/api/proxy', darkMode: false }
    if (patch) {
      setSettings(next)
    }
    persistToStorage(next)

    try {
      const response = await fetch(`${next.apiUrl}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          models_path: next.modelsPath,
          datasets_path: next.datasetsPath,
          output_path: next.outputPath,
          use_gpu: next.useGpu,
          gpu_device: next.gpuDevice,
        }),
        signal: AbortSignal.timeout(4000),
      })
      if (!response.ok) return false
      setLastSaved(new Date())
      return true
    } catch {
      return false
    } finally {
      setIsSaving(false)
    }
  }, [settings])

  return (
    <SettingsContext.Provider value={{ settings, updateSettings, saveSettings, isSaving, lastSaved }}>
      {children}
    </SettingsContext.Provider>
  )
}

export function useSettings() {
  const ctx = useContext(SettingsContext)
  if (!ctx) throw new Error('useSettings must be used inside SettingsProvider')
  return ctx
}
