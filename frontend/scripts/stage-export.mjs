// Stage the Next.js static export inside the Python package (postbuild hook).
//
// `next build` with output 'export' emits static assets into out/; the
// Xinference backend serves them from xinference/ui/web/dist so the wheel is
// self-contained. Standalone/dev builds produce no out/ directory, in which
// case there is nothing to stage.
import { cpSync, existsSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const frontendRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const exportDir = join(frontendRoot, 'out');
const destDir = join(frontendRoot, '..', 'xinference', 'ui', 'web', 'dist');

if (!existsSync(join(exportDir, 'index.html'))) {
  console.log(`[stage-export] no static export at ${exportDir}; skipping`);
  process.exit(0);
}

rmSync(destDir, { recursive: true, force: true });
cpSync(exportDir, destDir, { recursive: true });
console.log(`[stage-export] staged ${exportDir} -> ${destDir}`);
