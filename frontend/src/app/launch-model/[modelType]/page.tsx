import LaunchModelPageClient from './page-client';

// Server wrapper for static export: provides a placeholder param so a shell
// HTML is emitted for this dynamic route. The backend serves the shell for any
// real model type; the client component reads the actual values from the URL.
export function generateStaticParams() {
  return [{ modelType: '__shell__' }];
}

export default function LaunchModelPage() {
  return <LaunchModelPageClient />;
}
