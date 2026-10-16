/* eslint-disable @typescript-eslint/no-require-imports -- This Node preload hook uses CommonJS APIs. */
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const { transformSync } = require('next/dist/build/swc');

const frontendRoot = path.resolve(__dirname, '..');
const resolveFilename = Module._resolveFilename;

Module._resolveFilename = function (request, parent, isMain, options) {
  const resolvedRequest = request.startsWith('@/')
    ? path.join(frontendRoot, 'src', request.slice(2))
    : request;
  return resolveFilename.call(this, resolvedRequest, parent, isMain, options);
};

function compile(module, filename) {
  const extension = path.extname(filename);
  const { code } = transformSync(fs.readFileSync(filename, 'utf8'), {
    filename,
    jsc: {
      parser: {
        syntax: extension === '.mjs' ? 'ecmascript' : 'typescript',
        tsx: extension === '.tsx',
      },
      transform: {
        react: {
          runtime: 'automatic',
        },
      },
    },
    module: {
      type: 'commonjs',
    },
  });

  module._compile(code, filename);
}

require.extensions['.ts'] = compile;
require.extensions['.tsx'] = compile;
require.extensions['.mjs'] = compile;
