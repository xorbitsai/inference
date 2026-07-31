'use client';

import { useTheme } from 'next-themes';
import { Toaster as Sonner } from 'sonner';

type ToasterProps = React.ComponentProps<typeof Sonner>;

const Toaster = ({ ...props }: ToasterProps) => {
  const { theme } = useTheme();
  return (
    <Sonner
      theme={theme as ToasterProps['theme']}
      className="toaster group"
      position="top-center"
      expand
      visibleToasts={3}
      toastOptions={{
        classNames: {
          // Widen past sonner's 356px default: a server error can carry a full
          // Python traceback, which is unreadable at that width.
          toast:
            'group toast !w-[min(48rem,90vw)] group-[.toaster]:bg-background group-[.toaster]:text-foreground group-[.toaster]:border-border group-[.toaster]:shadow-lg data-[type=error]:!bg-red-500 data-[type=error]:!text-white data-[type=error]:!border-red-600 data-[type=success]:!bg-green-500 data-[type=success]:!text-white data-[type=success]:!border-green-500',
          description:
            'group-[.toast]:text-muted-foreground group-data-[type=error]:!text-white/90 group-data-[type=success]:!text-white/90',
          actionButton: 'group-[.toast]:bg-primary group-[.toast]:text-primary-foreground',
          cancelButton: 'group-[.toast]:bg-muted group-[.toast]:text-muted-foreground',
          icon: 'group-data-[type=error]:!text-white group-data-[type=success]:!text-white',
        },
      }}
      {...props}
    />
  );
};

export { Toaster };
