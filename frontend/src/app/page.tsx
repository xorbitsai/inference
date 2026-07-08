'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

// Client-side redirect: server redirect()/next.config redirects are not
// supported in static export.
export default function Home() {
  const router = useRouter();

  useEffect(() => {
    router.replace('/launch-model');
  }, [router]);

  return null;
}
