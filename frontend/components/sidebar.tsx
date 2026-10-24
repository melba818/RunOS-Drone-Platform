'use client'

import { useEffect } from 'react'
import { cn } from '@/lib/utils'
import { VisosLogo } from "@/components/ui/visos-logo"
import {
  Settings,
  Database,
  BarChart3,
  Images,
  ArrowUpDown,
  Copy,
  PenTool,
  Tag,
  Sparkles,
  Film,
  Scissors,
  ArrowLeftRight,
  GitMerge,
  FileCode,
  Cpu,
  Server,
  Bot,
  List,
  Activity,
  BarChart2,
  Archive,
  ScanSearch,
  FlaskConical,
  X,
  type LucideIcon,
} from 'lucide-react'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { ViewType, Dataset } from '@/app/page'

interface SidebarProps {
  activeView: ViewType
  setActiveView: (view: ViewType) => void
  selectedDataset: Dataset | null
  isOpen?: boolean
  embedded?: boolean
  onClose?: () => void
}

interface NavItem {
  id: ViewType
  label: string
  icon: LucideIcon
  description: string
}

interface NavGroup {
  label: string
  items: NavItem[]
}

const navGroups: NavGroup[] = [
  {
    label: '数据',
    items: [
      {
        id: 'datasets',
        label: '数据集',
        icon: Database,
        description: '加载并管理图像集合。可上传 ZIP 文件，或浏览本地文件夹。',
      },
      {
        id: 'dashboard',
        label: '数据看板',
        icon: BarChart3,
        description: '查看当前数据集的统计信息、类别分布与分析洞察。',
      },
      {
        id: 'gallery',
        label: '图库',
        icon: Images,
        description: '以可视化网格浏览全部图像，点击任意图像即可在标注器中打开。',
      },
      {
        id: 'sorting',
        label: '筛选排序',
        icon: ArrowUpDown,
        description: '逐张查看图像，并使用键盘快捷键快速整理数据。',
      },
      {
        id: 'duplicate-detection',
        label: '重复检测',
        icon: Copy,
        description: '自动查找并移除数据集中重复或高度相似的图像。',
      },
    ],
  },
  {
    label: '标注',
    items: [
      {
        id: 'annotate',
        label: '标注',
        icon: PenTool,
        description: '在图像上绘制边界框与标签，教 AI 识别目标对象。',
      },
      {
        id: 'classes',
        label: '类别',
        icon: Tag,
        description: '定义 AI 需要学习的目标类别（例如 “car”、“person”）。',
      },
    ],
  },
  {
    label: 'AI模型',
    items: [
      {
        id: 'compute-nodes',
        label: '算力节点',
        icon: Server,
        description: '管理 SSH 服务器，检测 GPU 与 Python 环境。云端大模型与远程训练都跑在这些机器上。',
      },
      {
        id: 'vlm',
        label: '大模型配置',
        icon: Bot,
        description: '接入视觉大模型（Qwen-VL / GPT-4o / 本地 vLLM、Ollama），配置接口与类别后用它做自动标注。',
      },
      {
        id: 'lmm',
        label: '多模态目标检测',
        icon: ScanSearch,
        description: '用自然语言描述目标（如「寻找水面上的船只」），由云端 VLX-Seek 检测器直接给出结论，并按规则告警。',
      },
    ],
  },
  {
    label: '处理',
    items: [
      {
        id: 'split',
        label: '训练 / 验证 / 测试',
        icon: Scissors,
        description: '将数据集划分为训练集、验证集与测试集，以便正确评估模型。',
      },
      {
        id: 'augmentation',
        label: '数据增强',
        icon: Sparkles,
        description: '自动应用翻转、旋转与颜色变化来扩充训练数据。',
      },
      {
        id: 'video-extraction',
        label: '视频抽帧',
        icon: Film,
        description: '从视频文件中提取帧图像，用作训练素材。',
      },
      {
        id: 'convert',
        label: '格式转换',
        icon: ArrowLeftRight,
        description: '在 YOLO、COCO、Pascal VOC 等格式之间转换数据集。',
      },
      {
        id: 'merge',
        label: '合并',
        icon: GitMerge,
        description: '将两个或多个独立数据集合并为一个统一数据集。',
      },
      {
        id: 'yaml-wizard',
        label: 'YAML 配置',
        icon: FileCode,
        description: '生成启动模型训练所需的配置文件。',
      },
    ],
  },
  {
    label: '训练',
    items: [
      {
        id: 'training',
        label: '模型训练',
        icon: Cpu,
        description: '在已标注的数据集上训练模型，实时跟踪 loss 与精度变化。',
      },
      {
        id: 'inference',
        label: '推理',
        icon: ScanSearch,
        description: '在图像、视频文件或实时摄像头上测试已训练模型，支持目标跟踪与 ReID。',
      },
      {
        id: 'evaluate',
        label: '评估',
        icon: FlaskConical,
        description: '多模型横向评估 — 对比 mAP、precision、recall 等指标。',
      },
      {
        id: 'models',
        label: '模型',
        icon: Bot,
        description: '查看与管理已训练的模型，可运行推理测试其效果。',
      },
      {
        id: 'batch-jobs',
        label: '批量任务',
        icon: List,
        description: '监控并管理导出、增强等长时间运行的后台任务。',
      },
    ],
  },
  {
    label: '分析',
    items: [
      {
        id: 'health',
        label: '健康检查',
        icon: Activity,
        description: '扫描数据集的质量问题，例如缺失标签或损坏的图像。',
      },
      {
        id: 'compare',
        label: '对比',
        icon: BarChart2,
        description: '对两个或多个数据集进行并排的统计对比。',
      },
      {
        id: 'snapshots',
        label: '快照',
        icon: Archive,
        description: '随时保存数据集快照，需要时可恢复。',
      },
    ],
  },
]

export function Sidebar({ activeView, setActiveView, selectedDataset, isOpen = false, embedded = false, onClose }: SidebarProps) {
  // Close sidebar on Escape key on mobile
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose?.() }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])

  const handleNavigate = (view: ViewType) => {
    setActiveView(view)
    onClose?.()
  }

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside className={cn(
        // 注意：不能在这里用 overflow-hidden —— 它会裁掉 Tooltip 浮层
        'relative flex flex-col bg-sidebar border-r border-sidebar-border select-none shrink-0',
        // Desktop: always visible, fixed width
        'md:w-56 md:h-full md:static md:translate-x-0 md:z-auto',
        embedded && 'pt-4',
        // Mobile: overlay slide-in
        'max-md:fixed max-md:inset-y-0 max-md:left-0 max-md:w-72 max-md:z-50 max-md:h-full max-md:transition-transform max-md:duration-300',
        isOpen ? 'max-md:translate-x-0' : 'max-md:-translate-x-full',
      )}>

      {/* Ambient glow */}
      <div className="ambient-orb w-48 h-48 bg-primary -top-16 -left-16 absolute overflow-hidden" />

      {/* Brand */}
      {(!embedded || isOpen) && (
        <div className="relative px-5 pt-5 pb-4 flex items-center justify-between">
          {!embedded && <VisosLogo size={36} showText={true} />}
          {/* Mobile close button */}
          {isOpen && (
            <button
              className="md:hidden p-1.5 rounded-lg text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-accent transition-colors"
              onClick={onClose}
              aria-label="关闭菜单"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      )}

      {/* Active dataset pill */}
      {selectedDataset && (
        <div className="mx-3 mb-3 px-3 py-2.5 rounded-xl border border-primary/20 bg-primary/8 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent pointer-events-none" />
          <p className="text-[9px] font-mono uppercase tracking-[0.18em] text-primary/60 mb-1">
            当前数据集
          </p>
          <p className="text-xs font-semibold text-sidebar-foreground truncate leading-tight">
            {selectedDataset.name}
          </p>
          <div className="flex items-center gap-2 mt-1.5">
            <span className="text-[9px] px-1.5 py-0.5 bg-primary/15 text-primary rounded-md font-mono tracking-wider border border-primary/20">
              {selectedDataset.format?.toUpperCase()}
            </span>
            <span className="text-[9px] font-mono text-muted-foreground">
              {selectedDataset.num_images?.toLocaleString()} 张图像
            </span>
          </div>
        </div>
      )}

      {/* Divider */}
      <div className="h-px mx-3 bg-sidebar-border mb-2" />

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-2 pb-2 space-y-4">
        {navGroups.map((group) => (
          <div key={group.label}>
            <p className="px-2 mb-1 text-[9px] font-mono font-semibold uppercase tracking-[0.2em] text-[#64748b] select-none">
              {group.label}
            </p>
            <div className="space-y-px">
              {group.items.map((item) => {
                const isActive = activeView === item.id
                const Icon = item.icon
                return (
                  <Tooltip key={item.id} delayDuration={700}>
                    <TooltipTrigger asChild>
                      <button
                        onClick={() => handleNavigate(item.id)}
                        disabled={process.env.NEXT_PUBLIC_RUNHIVE_PREVIEW === '1' && !['datasets', 'dashboard', 'gallery', 'classes', 'health'].includes(item.id)}
                        className={cn(
                          'group relative w-full flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-left',
                          'transition-all duration-150',
                          isActive
                            ? 'bg-primary/10 text-primary'
                            : 'text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent'
                        )}
                      >
                        {isActive && (
                          <span className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 bg-primary rounded-r-full" />
                        )}
                        <Icon
                          className={cn(
                            'w-3.5 h-3.5 shrink-0 transition-colors',
                            isActive
                              ? 'text-primary'
                              : 'text-muted-foreground/50 group-hover:text-sidebar-foreground/70'
                          )}
                          strokeWidth={1.75}
                        />
                        <span className={cn(
                          'text-[12px] leading-none transition-colors',
                          isActive ? 'font-semibold' : 'font-medium'
                        )}>
                          {item.label}{process.env.NEXT_PUBLIC_RUNHIVE_PREVIEW === '1' && !['datasets', 'dashboard', 'gallery', 'classes', 'health'].includes(item.id) ? '（未启用）' : ''}
                        </span>
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="right" sideOffset={10} className="max-w-[210px] py-2.5 px-3">
                      <p className="font-semibold text-[11px] mb-0.5">{item.label}</p>
                      <p className="text-[11px] text-surface-ink-foreground/70 leading-snug">{item.description}</p>
                    </TooltipContent>
                  </Tooltip>
                )
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* Footer — Settings */}
      <div className="px-2 py-2 border-t border-sidebar-border">
        <Tooltip delayDuration={700}>
          <TooltipTrigger asChild>
            <button
              onClick={() => handleNavigate('settings')}
              className={cn(
                'group relative w-full flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-left',
                'transition-all duration-150',
                activeView === 'settings'
                  ? 'bg-primary/10 text-primary'
                  : 'text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent'
              )}
            >
              {activeView === 'settings' && (
                <span className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 bg-primary rounded-r-full" />
              )}
              <Settings
                className={cn(
                  'w-3.5 h-3.5 shrink-0',
                  activeView === 'settings'
                    ? 'text-primary'
                    : 'text-muted-foreground/50 group-hover:text-sidebar-foreground/70'
                )}
                strokeWidth={1.75}
              />
              <span className={cn(
                'text-[12px] leading-none',
                activeView === 'settings' ? 'font-semibold' : 'font-medium'
              )}>
                设置
              </span>
            </button>
          </TooltipTrigger>
          <TooltipContent side="right" sideOffset={10} className="max-w-[210px] py-2.5 px-3">
            <p className="font-semibold text-[11px] mb-0.5">设置</p>
            <p className="text-[11px] text-surface-ink-foreground/70 leading-snug">配置 API 连接、文件路径、GPU 设备与外观。</p>
          </TooltipContent>
        </Tooltip>
      </div>
    </aside>
    </>
  )
}
