'use client';

import * as React from 'react';
import * as CollapsiblePrimitive from '@radix-ui/react-collapsible';
import { ChevronDown } from 'lucide-react';

import { cn } from '@/lib/utils';

function Collapsible({ ...props }: React.ComponentProps<typeof CollapsiblePrimitive.Root>) {
  return <CollapsiblePrimitive.Root data-slot="collapsible" {...props} />;
}

function CollapsibleTrigger({
  ...props
}: React.ComponentProps<typeof CollapsiblePrimitive.Trigger>) {
  return <CollapsiblePrimitive.Trigger data-slot="collapsible-trigger" {...props} />;
}

function CollapsibleContent({
  className,
  ...props
}: React.ComponentProps<typeof CollapsiblePrimitive.Content>) {
  return (
    <CollapsiblePrimitive.Content
      data-slot="collapsible-content"
      className={cn('collapsible-content overflow-hidden', className)}
      {...props}
    />
  );
}

interface CollapsiblePanelProps extends Omit<
  React.ComponentProps<typeof CollapsiblePrimitive.Root>,
  'title'
> {
  title: React.ReactNode;
  description?: React.ReactNode;
  icon?: React.ReactNode;
  contentClassName?: string;
}

function CollapsiblePanel({
  title,
  description,
  icon,
  children,
  className,
  contentClassName,
  ...props
}: CollapsiblePanelProps) {
  return (
    <Collapsible
      className={cn('group overflow-hidden rounded-xl border bg-card duration-300', className)}
      {...props}
    >
      <CollapsibleTrigger className="flex w-full items-center gap-3 px-4 py-3 text-left outline-none transition-colors duration-300 hover:bg-accent/60 ">
        {icon && (
          // <span className="shrink-0 flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
          <span className="shrink-0">{icon}</span>
        )}
        <span className="min-w-0 flex-1">
          <span className="block font-medium text-foreground">{title}</span>
          {description && (
            <span className="mt-1 block text-sm leading-5 text-muted-foreground">
              {description}
            </span>
          )}
        </span>
        <span className="flex shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors duration-300">
          <ChevronDown className="size-4 transition-transform duration-300 ease-out group-data-[state=open]:rotate-180 motion-reduce:transition-none" />
        </span>
      </CollapsibleTrigger>
      <CollapsibleContent forceMount>
        <div className={cn('border-t border-border/70 p-4 text-sm leading-6', contentClassName)}>
          {children}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

export { Collapsible, CollapsibleContent, CollapsiblePanel, CollapsibleTrigger };
