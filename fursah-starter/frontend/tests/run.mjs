// Use the existing Vite compiler and Node test runner; no added dependencies.
import { build } from 'vite';
import { mkdtemp, rm } from 'node:fs/promises';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

const temporary = await mkdtemp(resolve('node_modules/.fursah-tests-'));
try {
  await build({
    configFile: false, envDir: temporary, logLevel: 'warn',
    esbuild: { jsx: 'automatic' },
    build: {
      ssr: resolve('tests/mvp.test.tsx'), outDir: temporary, emptyOutDir: false,
      rollupOptions: { output: { entryFileNames: 'tests.mjs' } },
    },
  });
  const result = spawnSync(process.execPath, ['--test', resolve(temporary, 'tests.mjs')], { stdio: 'inherit' });
  process.exitCode = result.status ?? 1;
} finally {
  await rm(temporary, { recursive: true, force: true });
}
