import * as React from "react";

import { cn } from "@/lib/utils";

interface InputProps
  extends React.ComponentProps<"input"> {
  error?: boolean;
}

function Input({
  className,
  type,
  error,
  ...props
}: InputProps) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "border-input flex h-9 w-full rounded-md border bg-transparent px-3 py-1 text-sm outline-none transition-all",

        "focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]",

        error &&
          "border-destructive focus-visible:border-destructive focus-visible:ring-destructive/40",

        className
      )}
      placeholder="Please enter"
      {...props}
    />
  );
}

export { Input };