type ModelCapability = { type?: string; supports_confidence_threshold?: boolean }

export function supportsConfidenceThreshold(model?: ModelCapability): boolean {
  if (!model) return false
  if (typeof model.supports_confidence_threshold === 'boolean') return model.supports_confidence_threshold
  return !['vlm', 'vlx', 'vlx_qwen'].includes((model.type || '').toLowerCase())
}

export function hasDetectorConfidence(result: { confidence?: number; confidence_source?: string }): boolean {
  return (result.confidence_source === undefined || result.confidence_source === 'detector')
    && typeof result.confidence === 'number' && Number.isFinite(result.confidence)
}
