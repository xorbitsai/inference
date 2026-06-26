import type { CodeExampleConfig, CodeExampleField } from './types';

type CodeLanguage = 'python' | 'typescript' | 'java' | 'go' | 'shell';

export const CODE_LANGUAGE_OPTIONS: { label: string; value: CodeLanguage; highlight: string }[] = [
  { label: 'Python', value: 'python', highlight: 'python' },
  { label: 'TypeScript', value: 'typescript', highlight: 'typescript' },
  { label: 'Java', value: 'java', highlight: 'java' },
  { label: 'Go', value: 'go', highlight: 'go' },
  { label: 'Shell', value: 'shell', highlight: 'bash' },
];

export const CHAT_CODE_EXAMPLE: CodeExampleConfig = {
  method: 'POST',
  contentType: 'json',
  fields: [
    { key: 'model', required: true },
    {
      key: 'messages',
      required: true,
      value: [{ role: 'user', content: 'Hello, what can you do?' }],
    },
    { key: 'stream', value: false, comment: 'Optional' },
    { key: 'max_tokens', value: 4000 },
    { key: 'top_p', value: 1 },
    { key: 'top_k', value: 40 },
    { key: 'presence_penalty', value: 0 },
    { key: 'frequency_penalty', value: 0 },
    { key: 'temperature', value: 0.6 },
  ],
};

const DEFAULT_VALUES: Record<string, unknown> = {
  model: '{MODEL_UID}',
  prompt: 'Hello, what can you do?',
  input: 'Hello, can you read this text aloud?',
  voice: 'default',
  image: '/path/to/image.png',
  mask_image: '/path/to/mask.png',
  first_frame: '/path/to/first-frame.png',
  last_frame: '/path/to/last-frame.png',
  file: './audio.wav',
  negative_prompt: '',
  language: 'en',
  response_format: 'json',
  speed: 1,
  n: 1,
  size: '1024*1024',
  stream: false,
  temperature: 1,
  max_tokens: 256,
  kwargs: {},
};

function resolveValue(field: CodeExampleField, modelUid: string) {
  if (field.key === 'model') {
    return modelUid || DEFAULT_VALUES.model;
  }

  if (field.value !== undefined) {
    return field.value;
  }

  if (field.key in DEFAULT_VALUES) {
    return DEFAULT_VALUES[field.key];
  }

  return '';
}

function fieldComment(field: CodeExampleField, prefix: string) {
  if (field.required && !field.comment) return '';
  return ` ${prefix} ${field.comment || 'Optional'}`;
}

function escapeString(value: string) {
  return value.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

function pythonLiteral(value: unknown): string {
  if (typeof value === 'string') return `"${escapeString(value)}"`;
  if (typeof value === 'number') return String(value);
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  if (value === null) return 'None';
  if (Array.isArray(value)) {
    return `[${value.map((item) => pythonLiteral(item)).join(', ')}]`;
  }
  if (typeof value === 'object') {
    return `{${Object.entries(value as Record<string, unknown>)
      .map(([key, item]) => `"${key}": ${pythonLiteral(item)}`)
      .join(', ')}}`;
  }
  return 'None';
}

function jsLiteral(value: unknown): string {
  return JSON.stringify(value, null, 2)
    .replace(/"([^"]+)":/g, '$1:')
    .replace(/\n/g, '\n  ');
}

function jsonValue(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function shellValue(value: unknown): string {
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function javaValue(value: unknown): string {
  if (typeof value === 'string') return `"${escapeString(value)}"`;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (value === null) return 'null';
  return `objectMapper.writeValueAsString(objectMapper.readValue("""
${jsonValue(value)}
""", Object.class))`;
}

function goValue(value: unknown): string {
  if (typeof value === 'string') return `"${escapeString(value)}"`;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (value === null) return 'nil';
  return `json.RawMessage(\`${jsonValue(value)}\`)`;
}

function pyFieldValue(field: CodeExampleField, modelUid: string) {
  const value = resolveValue(field, modelUid);
  if (field.stringify) {
    return `json.dumps(${pythonLiteral(value)})`;
  }
  return pythonLiteral(value);
}

function tsFieldValue(field: CodeExampleField, modelUid: string) {
  const value = resolveValue(field, modelUid);
  if (field.stringify) {
    return `JSON.stringify(${jsLiteral(value)})`;
  }
  return jsLiteral(value);
}

function generatePython(config: CodeExampleConfig, url: string, modelUid: string) {
  const fields = config.fields;
  const isForm = config.contentType === 'form';
  const imports = ['import requests'];
  if (fields.some((field) => field.stringify)) {
    imports.push('import json');
  }

  const lines = [...imports, '', `url = "${url}"`, ''];

  if (isForm) {
    const dataFields = fields.filter((field) => field.type !== 'file');
    const fileFields = fields.filter((field) => field.type === 'file');

    lines.push('data = {');
    dataFields.forEach((field) => {
      lines.push(
        `    "${field.key}": ${pyFieldValue(field, modelUid)},${fieldComment(field, '#')}`
      );
    });
    lines.push('}', '');

    if (fileFields.length) {
      lines.push('files = {');
      fileFields.forEach((field) => {
        const value = resolveValue(field, modelUid);
        lines.push(
          `    "${field.key}": open("${escapeString(String(value))}", "rb"),${fieldComment(field, '#')}`
        );
      });
      lines.push('}', '');
    }
  } else {
    lines.push('data = {');
    fields.forEach((field) => {
      lines.push(
        `    "${field.key}": ${pyFieldValue(field, modelUid)},${fieldComment(field, '#')}`
      );
    });
    lines.push('}', '');
  }

  lines.push('headers = {');
  lines.push('    "Accept": "application/json",');
  if (!isForm) {
    lines.push('    "Content-Type": "application/json",');
  }
  lines.push('    "Authorization": "Bearer {API_KEY}",');
  lines.push('}', '');
  lines.push(
    `requests.${config.method.toLowerCase()}(url, headers=headers, ${isForm ? 'data=data, files=files' : 'json=data'})`
  );

  return lines.join('\n');
}

function generateTS(config: CodeExampleConfig, url: string, modelUid: string) {
  const isForm = config.contentType === 'form';
  const lines = [`const url = "${url}";`, ''];

  if (isForm) {
    lines.push('const formData = new FormData();');
    config.fields.forEach((field) => {
      const value = resolveValue(field, modelUid);
      const resolvedValue =
        field.type === 'file' ? `${field.key}File` : tsFieldValue(field, modelUid);
      if (field.type === 'file') {
        lines.push(
          `// const ${field.key}File = await fileFromPath("${escapeString(String(value))}");`
        );
      }
      lines.push(`formData.append("${field.key}", ${resolvedValue});${fieldComment(field, '//')}`);
    });
  } else {
    lines.push('const data = {');
    config.fields.forEach((field) => {
      lines.push(`  ${field.key}: ${tsFieldValue(field, modelUid)},${fieldComment(field, '//')}`);
    });
    lines.push('};');
  }

  lines.push('', 'await fetch(url, {');
  lines.push(`  method: "${config.method}",`);
  lines.push('  headers: {');
  lines.push('    Accept: "application/json",');
  if (!isForm) {
    lines.push('    "Content-Type": "application/json",');
  }
  lines.push('    Authorization: "Bearer {API_KEY}",');
  lines.push('  },');
  lines.push(`  body: ${isForm ? 'formData' : 'JSON.stringify(data)'},`);
  lines.push('});');

  return lines.join('\n');
}

function generateJava(config: CodeExampleConfig, url: string, modelUid: string) {
  if (config.contentType === 'form') {
    const lines = [
      'import java.io.File;',
      'import okhttp3.*;',
      '',
      'OkHttpClient client = new OkHttpClient();',
      '',
      'MultipartBody.Builder formData = new MultipartBody.Builder().setType(MultipartBody.FORM);',
    ];

    config.fields.forEach((field) => {
      const value = resolveValue(field, modelUid);
      if (field.type === 'file') {
        lines.push(
          `formData.addFormDataPart("${field.key}", "${escapeString(String(value))}", RequestBody.create(new File("${escapeString(String(value))}"), MediaType.parse("application/octet-stream")));${fieldComment(field, '//')}`
        );
      } else {
        lines.push(
          `formData.addFormDataPart("${field.key}", String.valueOf(${javaValue(value)}));${fieldComment(field, '//')}`
        );
      }
    });

    lines.push(
      '',
      `Request request = new Request.Builder()`,
      `    .url("${url}")`,
      '    .addHeader("Accept", "application/json")',
      '    .addHeader("Authorization", "Bearer {API_KEY}")',
      '    .post(formData.build())',
      '    .build();',
      '',
      'Response response = client.newCall(request).execute();'
    );
    return lines.join('\n');
  }

  const lines = [
    'import com.fasterxml.jackson.databind.ObjectMapper;',
    'import java.net.URI;',
    'import java.net.http.HttpClient;',
    'import java.net.http.HttpRequest;',
    'import java.net.http.HttpResponse;',
    'import java.util.LinkedHashMap;',
    'import java.util.Map;',
    '',
    'ObjectMapper objectMapper = new ObjectMapper();',
    'Map<String, Object> data = new LinkedHashMap<>();',
  ];

  config.fields.forEach((field) => {
    lines.push(
      `data.put("${field.key}", ${field.stringify ? javaValue(resolveValue(field, modelUid)) : javaValue(resolveValue(field, modelUid))});${fieldComment(field, '//')}`
    );
  });

  lines.push(
    '',
    'String body = objectMapper.writeValueAsString(data);',
    '',
    'HttpRequest request = HttpRequest.newBuilder()',
    `    .uri(URI.create("${url}"))`,
    '    .header("Accept", "application/json")',
    '    .header("Content-Type", "application/json")',
    '    .header("Authorization", "Bearer {API_KEY}")',
    '    .POST(HttpRequest.BodyPublishers.ofString(body))',
    '    .build();',
    '',
    'HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString());'
  );

  return lines.join('\n');
}

function generateGo(config: CodeExampleConfig, url: string, modelUid: string) {
  if (config.contentType === 'form') {
    const lines = [
      'package main',
      '',
      'import (',
      '  "bytes"',
      '  "io"',
      '  "mime/multipart"',
      '  "net/http"',
      '  "os"',
      ')',
      '',
      'func main() {',
      '  body := &bytes.Buffer{}',
      '  writer := multipart.NewWriter(body)',
    ];

    config.fields.forEach((field) => {
      const value = resolveValue(field, modelUid);
      if (field.type === 'file') {
        lines.push(
          `  file, _ := os.Open("${escapeString(String(value))}")${fieldComment(field, '//')}`,
          '  defer file.Close()',
          `  part, _ := writer.CreateFormFile("${field.key}", "${escapeString(String(value))}")`,
          '  io.Copy(part, file)'
        );
      } else {
        lines.push(
          `  writer.WriteField("${field.key}", ${goValue(value)})${fieldComment(field, '//')}`
        );
      }
    });

    lines.push(
      '  writer.Close()',
      '',
      `  req, _ := http.NewRequest("${config.method}", "${url}", body)`,
      '  req.Header.Set("Accept", "application/json")',
      '  req.Header.Set("Content-Type", writer.FormDataContentType())',
      '  req.Header.Set("Authorization", "Bearer {API_KEY}")',
      '',
      '  http.DefaultClient.Do(req)',
      '}'
    );
    return lines.join('\n');
  }

  const lines = [
    'package main',
    '',
    'import (',
    '  "bytes"',
    '  "encoding/json"',
    '  "net/http"',
    ')',
    '',
    'func main() {',
    '  data := map[string]interface{}{',
  ];

  config.fields.forEach((field) => {
    lines.push(
      `    "${field.key}": ${goValue(resolveValue(field, modelUid))},${fieldComment(field, '//')}`
    );
  });

  lines.push(
    '  }',
    '',
    '  body, _ := json.Marshal(data)',
    `  req, _ := http.NewRequest("${config.method}", "${url}", bytes.NewBuffer(body))`,
    '  req.Header.Set("Accept", "application/json")',
    '  req.Header.Set("Content-Type", "application/json")',
    '  req.Header.Set("Authorization", "Bearer {API_KEY}")',
    '',
    '  http.DefaultClient.Do(req)',
    '}'
  );

  return lines.join('\n');
}

function generateShell(config: CodeExampleConfig, url: string, modelUid: string) {
  const requiredFields = config.fields.filter((field) => field.required);
  const lines = [
    `curl -X ${config.method} "${url}"`,
    `  -H "Accept: application/json"`,
    `  -H "Authorization: Bearer {API_KEY}"`,
  ];

  if (config.contentType === 'json') {
    const body = requiredFields.reduce<Record<string, unknown>>((result, field) => {
      result[field.key] = resolveValue(field, modelUid);
      return result;
    }, {});
    lines.push('  -H "Content-Type: application/json"');
    lines.push(`  -d '${JSON.stringify(body, null, 2)}'`);
  } else {
    requiredFields.forEach((field) => {
      const value = resolveValue(field, modelUid);
      lines.push(
        `  -F "${field.key}=${field.type === 'file' ? `@${shellValue(value)}` : shellValue(value)}"`
      );
    });
  }

  return lines.map((line, index) => (index === lines.length - 1 ? line : `${line} \\`)).join('\n');
}

export function generateCodeExample(
  language: CodeLanguage,
  config: CodeExampleConfig,
  url: string,
  modelUid: string
) {
  switch (language) {
    case 'python':
      return generatePython(config, url, modelUid);
    case 'typescript':
      return generateTS(config, url, modelUid);
    case 'java':
      return generateJava(config, url, modelUid);
    case 'go':
      return generateGo(config, url, modelUid);
    case 'shell':
      return generateShell(config, url, modelUid);
  }
}
