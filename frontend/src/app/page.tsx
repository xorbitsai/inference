// "use client";

// import { useGlobal } from '@/contexts/global-context';
// import { Loader2 } from 'lucide-react';

// export default function Home() {
//   const { globalReady } = useGlobal();
//   if(!globalReady) {
//     return (
//       <div className="h-screen w-screen flex items-center justify-center bg-background">
//         <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
//       </div>
//     )
//   }
//   return null;
// }
// next.config: redirect root '/' to '/workbench'
export default function Home() {
  return null;
}