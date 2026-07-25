// Stage the Next.js static export inside the Python package (postbuild hook).
//
// `next build` with output 'export' emits static assets into out/; the
// Xinference backend serves them from xinference/ui/web/dist so the wheel is
// self-contained. Standalone/dev builds produce no out/ directory, in which
// case there is nothing to stage.
import { cpSync, existsSync, renameSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const frontendRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const exportDir = join(frontendRoot, 'out');
const destDir = join(frontendRoot, '..', 'xinference', 'ui', 'web', 'dist');

if (!existsSync(join(exportDir, 'index.html'))) {
  console.log(`[stage-export] no static export at ${exportDir}; skipping`);
  process.exit(0);
}

// Stage into a temporary directory first, then atomically rename so the
// destination directory only changes when the full export tree is in place.
// This guarantees that the Python-side mtime-based hot-reload never scans a
// partially copied tree.
const stagingDir = destDir + '.staging';
rmSync(stagingDir, { recursive: true, force: true });
cpSync(exportDir, stagingDir, { recursive: true });
rmSync(destDir, { recursive: true, force: true });
renameSync(stagingDir, destDir);
console.log(`[stage-export] staged ${exportDir} -> ${destDir}`);
