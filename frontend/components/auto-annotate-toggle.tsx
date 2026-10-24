import { cn } from '@/lib/utils'

type Props = { label: string; enabled: boolean; disabled?: boolean; onToggle: () => void }

export function AutoAnnotateToggle({ label, enabled, disabled, onToggle }: Props) {
  return <div className="flex items-center justify-between gap-2">
    <p className="text-[9px] font-mono font-semibold uppercase tracking-[0.18em] text-primary/70">{label}</p>
    <button type="button" aria-label={`${label}：${enabled ? '已启用' : '关闭'}`} aria-pressed={enabled}
      disabled={disabled} onClick={onToggle}
      className={cn('px-1.5 py-0.5 rounded text-[9px] font-mono transition-colors border shrink-0',
        enabled ? 'bg-primary/15 text-primary border-primary/30'
          : 'text-muted-foreground/60 border-border hover:text-primary hover:border-primary/40')}
    >{enabled ? '已启用' : '关闭'}</button>
  </div>
}
