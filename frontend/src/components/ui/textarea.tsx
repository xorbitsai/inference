import * as React from "react";

import { cn } from "@/lib/utils";

interface TextareaProps
  extends React.ComponentProps<"textarea"> {
  error?: boolean;
}

const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  TextareaProps
>(({ className, error, ...props }, ref) => {
  return (
    <textarea
      ref={ref}
      data-slot="textarea"
      className={cn(
        "border-input flex min-h-20 w-full rounded-md border bg-transparent px-3 py-2 text-sm outline-none transition-all resize-none",

        "focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]",

        error &&
          "border-destructive focus-visible:border-destructive focus-visible:ring-destructive/40",

        className
      )}
      placeholder="Please enter"
      {...props}
    />
  );
});

Textarea.displayName = "Textarea";

export { Textarea };