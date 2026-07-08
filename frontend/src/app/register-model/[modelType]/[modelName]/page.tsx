import RegisterModelEditPageClient from './page-client';

// Server wrapper for static export: provides placeholder params so a shell
// HTML is emitted for this dynamic route. The backend serves the shell for any
// real model type/name; the client component reads the actual values from the
// URL.
export function generateStaticParams() {
  return [{ modelType: '__shell__', modelName: '__shell__' }];
}

export default function RegisterModelEditPage() {
  return <RegisterModelEditPageClient />;
}
