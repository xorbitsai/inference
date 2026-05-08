'use client';

// import { usePathname } from "next/navigation";
// import { Sidebar } from "@/components/layout/sidebar";

interface LayoutContentProps {
  children: React.ReactNode;
}

export function LayoutContent({ children }: LayoutContentProps) {
  return (
    <div className="flex h-screen bg-background relative">
      {/* <Sidebar /> */}
      <main className="flex-1 flex flex-col overflow-hidden bg-background">{children}</main>
    </div>
  );
}
