/** 「多模态目标检测」与「推理 · 视频 + 大模型」共用的文案常量。
 *
 *  两边必须是**同一份**：用户在哪个页面看到的目标描述句式、任务类型与说明，
 *  行为就该完全一致（同一条云端 VLX-Seek 链路）。放在这里避免各自复制一份后走偏。
 */

/** 常用目标描述。点击只负责把文字填进输入框，**不做任何解析**。

    刻意覆盖不同场景（地面 / 道路 / 水域 / 行为 / 穿戴属性），不要清一色写"水面上的"——
    用户一眼看到就能挑到接近自己任务的句式。

    ★不带「寻找」前缀（2026-09-26 起）★：这是一栏「目标描述」，标签本身就是目标，
    前缀会让五个标签的**目标词全挤在同一个位置**，一眼扫过去分不清差别。
    VLX-Seek 吃的是这句话本身，去掉前缀不影响它理解。

    但**仍要写成完整的名词短语**：实测单字「人」会把船认成人，写成「人员」才稳 ——
    所以别为了短把标签缩到一两个字的裸词。 */
export const PROMPT_PRESETS = [
  '人员',
  '道路上行驶的车辆',
  '水面上的船只',
  '野泳人员',
  '水面上未穿救生衣的人员',
]

/** VLX-Seek 任务类型。detection 实测最稳；reasoning_detection 带思维链但更慢更易复读。
 *
 * 两种都**不使用关键词解析**：用户写什么句子，就原样把这句话发过去，
 * 区别只在模型那侧的作答方式。 */
export const VLX_TASKS = [
  { value: 'detection', label: 'detection' },
  { value: 'reasoning_detection', label: 'reasoning_detection' },
  { value: 'grounding_single', label: 'grounding_single' },
  { value: 'counting', label: 'counting' },
]

/** 选中某一项后在下方给出的**一句话**说明（不放进下拉选项里，避免选项名变长）。
 *  用户明确要求保留这一行，别删。 */
export const VLX_TASK_DESC: Record<string, string> = {
  detection: '普通目标检测，适合「人员」「船只」这类直接的目标名称，速度最快。',
  reasoning_detection: '复杂语义检测，适合「未穿救生衣」「远离船只」这类带限定的描述，会先推理再作答，明显更慢。',
  grounding_single: '只定位一个目标，画面里取最匹配的那个。',
  counting: '计数为主，返回数量比框更重要。',
}

/** Saved VLX connection metadata shown by older server builds that predate
 * GET /api/lmm/vlx/config. This is display metadata only: no credentials or
 * health state, and it never starts or contacts the model service. The live
 * backend response takes precedence whenever that route is available. */
export const VLX_PROVIDER_ID = 'vlx-seek'
export const SAVED_VLX_PROVIDER = {
  config_id: VLX_PROVIDER_ID,
  display_name: 'VLX-Seek',
  model: 'VLX-Seek-1.5-10B',
  kind_label: '专用视觉模型',
}

export interface VlxConnectionSummary {
  configured: boolean
  enabled: boolean
  transport: 'direct' | 'ssh'
  node_id: string
  port: number
  base_url: string
  token_configured: boolean
  timeout?: number
  task?: 'detection' | 'reasoning_detection' | 'grounding_single' | 'counting'
  lang?: 'zh' | 'en'
  max_new_tokens?: number
  frame_interval?: number
  extra_instruction?: string
  supports_persistent_prompt?: boolean
}

// Sanitized values verified in the A6000 deployment's saved lmm_config.json.
// enabled remains false; this only lets the UI show the already-saved entry.
const LEGACY_SAVED_VLX_SUMMARY: VlxConnectionSummary = {
  configured: true,
  enabled: false,
  transport: 'direct',
  node_id: '',
  port: 8021,
  base_url: 'http://127.0.0.1:8021',
  token_configured: true,
  timeout: 300,
  task: 'detection',
  lang: 'zh',
  max_new_tokens: 256,
  frame_interval: 1,
  extra_instruction: '',
}

/** Read only saved connection metadata. Falls back only for the legacy server
 * whose backend has no summary route yet. This is never a model probe. */
export async function readVlxConnectionSummary(apiBase: string): Promise<VlxConnectionSummary> {
  try {
    const response = await fetch(`${apiBase}/api/lmm/vlx/config`, { cache: 'no-store' })
    if (response.status === 404) return { ...LEGACY_SAVED_VLX_SUMMARY }
    if (!response.ok) return { ...LEGACY_SAVED_VLX_SUMMARY, configured: false, token_configured: false }
    const data = await response.json()
    if (data?.config && typeof data.config.configured === 'boolean') {
      return {
        ...data.config,
        supports_persistent_prompt: Object.prototype.hasOwnProperty.call(data.config, 'extra_instruction'),
      } as VlxConnectionSummary
    }
  } catch {
    // The static metadata keeps the configured server entry visible offline.
  }
  return { ...LEGACY_SAVED_VLX_SUMMARY }
}

/** 云端 VLX-Seek 常驻服务状态（GET /api/lmm/vlx/health）。
 *
 *  两个页面（多模态目标检测 / 推理·视频+大模型）显示的是**同一份**状态：
 *  同一条云端链路，一边说已连接另一边说未连接就是 bug。类型也共用，避免字段走偏。 */
export interface VlxHealth {
  connected: boolean
  status?: string
  model?: string
  gpu?: string
  device?: string
  load_seconds?: number | null
  vram_free_mb?: number
  vram_total_mb?: number
  load_error?: string | null
  message?: string
  runner?: Record<string, unknown>
}
