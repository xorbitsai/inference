import { CircleCheck, Settings2, ShieldCheck, SlidersHorizontal, Sparkles } from 'lucide-react';

import { CollapsiblePanel } from '@/components/ui/collapsible';

export default function Page() {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-8 py-8">
      <div>
        <div className="mb-3 flex items-center gap-2 text-sm font-medium text-primary">
          <Sparkles className="size-4" />
          配置中心
        </div>
        <h1 className="text-3xl font-semibold tracking-tight">模型服务设置</h1>
        <p className="mt-3 max-w-2xl leading-6 text-muted-foreground">
          按需展开配置项。常用选项保持在前面，进阶内容收起后页面会更清爽。
        </p>
      </div>

      <div className="flex flex-col gap-3">
        <CollapsiblePanel
          defaultOpen
          icon={<Settings2 className="size-4" />}
          title="基础配置"
          description="设置模型名称、类型和运行方式"
        >
          <div className="grid gap-3 sm:grid-cols-2">
            <SettingItem label="模型名称" value="Qwen2.5-Instruct" />
            <SettingItem label="运行方式" value="本地推理" />
          </div>
        </CollapsiblePanel>

        <CollapsiblePanel
          icon={<SlidersHorizontal className="size-4" />}
          title="性能与资源"
          description="调整设备、并发和缓存策略"
        >
          <div className="grid gap-3 sm:grid-cols-2">
            <SettingItem label="运行设备" value="自动选择" />
            <SettingItem label="最大并发" value="8 个请求" />
          </div>
        </CollapsiblePanel>

        <CollapsiblePanel
          icon={<ShieldCheck className="size-4" />}
          title="访问控制"
          // description="管理服务可见范围和接口权限"
        >
          <div className="flex items-start gap-3">
            <CircleCheck className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>当前服务仅对工作区成员开放。你可以在启动服务后继续配置 API 密钥和访问策略。</p>
          </div>
        </CollapsiblePanel>
      </div>
    </div>
  );
}

function SettingItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-muted/60 px-4 py-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 font-medium text-foreground">{value}</p>
    </div>
  );
}
