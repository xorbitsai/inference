'use client';

import { useMemo, useState } from 'react';
import { Copy, ExternalLink, X } from 'lucide-react';
import { useRouter } from 'next/navigation';

import { Button } from '@/components/ui/button';
import { JSONSyntaxHighlighter } from '@/components/ui/json-syntax-highlighter';
import { Sheet, SheetClose, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ModelAbility } from '@/constants';
import {
  CHAT_CODE_EXAMPLE,
  CODE_EXAMPLE_DEFAULT_VALUES,
  CODE_LANGUAGE_OPTIONS,
  type CodeExampleConfig,
  type CodeExampleField,
  type CodeLanguage,
} from '@/constants/running';
import { useMenuAuth } from '@/hooks/use-menu-auth';
import { copyText, getApiUrl } from '@/lib/utils';

import { CAPABILITY_CONFIGS } from '../capability-config';

interface TryApiDrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  modelUid?: string;
  ability?: ModelAbility;
}

function resolveValue(field: CodeExampleField, modelUid: string) {
  if (field.key === 'model') {
    return modelUid || CODE_EXAMPLE_DEFAULT_VALUES.model;
  }

  if (field.value !== undefined) {
    return field.value;
  }

  if (field.key in CODE_EXAMPLE_DEFAULT_VALUES) {
    return CODE_EXAMPLE_DEFAULT_VALUES[field.key];
  }

  return '';
}

function fieldComment(field: CodeExampleField, prefix: string) {
  if (field.required && !field.comment) return '';
  return ` ${prefix} ${field.comment || 'Optional'}`;
}

function escapeString(value: string) {
  return value
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t');
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

function compactJsonValue(value: unknown): string {
  return JSON.stringify(value) ?? '';
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

function shouldStringifyFormValue(field: CodeExampleField, value: unknown) {
  return field.stringify || (typeof value === 'object' && value !== null);
}

function formStringValue(field: CodeExampleField, value: unknown) {
  if (shouldStringifyFormValue(field, value)) {
    return compactJsonValue(value);
  }
  if (value === null) {
    return '';
  }
  return String(value);
}

function javaFormValue(field: CodeExampleField, value: unknown): string {
  return `"${escapeString(formStringValue(field, value))}"`;
}

function goFormValue(field: CodeExampleField, value: unknown): string {
  return `"${escapeString(formStringValue(field, value))}"`;
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
        lines.push(`const ${field.key}File = new File([""], "${escapeString(String(value))}");`);
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
          `formData.addFormDataPart("${field.key}", ${javaFormValue(field, value)});${fieldComment(field, '//')}`
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
      `data.put("${field.key}", ${javaValue(resolveValue(field, modelUid))});${fieldComment(field, '//')}`
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
        const key = field.key;
        lines.push(
          `  file_${key}, _ := os.Open("${escapeString(String(value))}")${fieldComment(field, '//')}`,
          `  defer file_${key}.Close()`,
          `  part_${key}, _ := writer.CreateFormFile("${field.key}", "${escapeString(String(value))}")`,
          `  io.Copy(part_${key}, file_${key})`
        );
      } else {
        lines.push(
          `  writer.WriteField("${field.key}", ${goFormValue(field, value)})${fieldComment(field, '//')}`
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

function generateCodeExample(
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

function getCodeExample(
  ability?: ModelAbility
): { requestApi: string; config: CodeExampleConfig } | null {
  if (ability === ModelAbility.Chat) {
    return {
      requestApi: '/v1/chat/completions',
      config: CHAT_CODE_EXAMPLE,
    };
  }

  if (!ability) return null;

  const capabilityConfig = CAPABILITY_CONFIGS[ability];
  if (!capabilityConfig?.codeExample) return null;

  return {
    requestApi: capabilityConfig.requestApi,
    config: capabilityConfig.codeExample,
  };
}

export function getTryApiAbility(abilities: ModelAbility[] = []) {
  const primaryAbilities = abilities.filter((ability) => !ability.includes('_'));
  return primaryAbilities.find((ability) => ability === ModelAbility.Chat) || primaryAbilities[0];
}

export function TryApiDrawer({
  open,
  onOpenChange,
  modelUid = '{MODEL_UID}',
  ability,
}: TryApiDrawerProps) {
  const router = useRouter();
  const { canAccessKeysPage } = useMenuAuth();
  const [language, setLanguage] = useState(CODE_LANGUAGE_OPTIONS[0].value);
  const codeExample = getCodeExample(ability);
  const url = codeExample ? `${getApiUrl()}${codeExample.requestApi}` : '';
  const code = useMemo(() => {
    if (!codeExample) return '';
    return generateCodeExample(language, codeExample.config, url, modelUid);
  }, [codeExample, language, modelUid, url]);
  const activeLanguage = CODE_LANGUAGE_OPTIONS.find((item) => item.value === language);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent showClose={false} className="w-[min(50vw,760px)] gap-0 p-0 sm:max-w-none">
        <SheetHeader className="flex-row items-center justify-between gap-4 border-b px-6 py-5">
          <div className="flex min-w-0 items-center gap-3">
            <SheetClose asChild>
              <Button
                variant="ghost"
                size="icon"
                className="size-9 rounded-full text-muted-foreground"
              >
                <X className="size-5" />
              </Button>
            </SheetClose>
            <SheetTitle className="truncate text-xl">Try To API</SheetTitle>
          </div>
          <Button
            type="button"
            variant="outline"
            disabled={!canAccessKeysPage}
            onClick={() => router.push('/api-key-management')}
          >
            <ExternalLink className="size-4" />
            Get API Key
          </Button>
        </SheetHeader>
        <div className="flex-1 overflow-y-auto p-6">
          {codeExample ? (
            <Tabs value={language} onValueChange={(value) => setLanguage(value as typeof language)}>
              <TabsList className="grid h-11 w-full grid-cols-5 rounded-xl p-1">
                {CODE_LANGUAGE_OPTIONS.map((item) => (
                  <TabsTrigger
                    key={item.value}
                    value={item.value}
                    className="p-0 rounded-md text-base data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
                  >
                    {item.label}
                  </TabsTrigger>
                ))}
              </TabsList>
              {CODE_LANGUAGE_OPTIONS.map((item) => (
                <TabsContent key={item.value} value={item.value} className="mt-6">
                  <div className="relative">
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="absolute right-3 top-3 z-10 size-8 text-muted-foreground"
                      onClick={() => copyText(code)}
                    >
                      <Copy className="size-4" />
                    </Button>
                    <JSONSyntaxHighlighter
                      code={code}
                      language={activeLanguage?.highlight || 'python'}
                      className="rounded-xl pr-12 text-base whitespace-pre-wrap break-all"
                    />
                  </div>
                </TabsContent>
              ))}
            </Tabs>
          ) : (
            <div className="flex min-h-80 items-center justify-center rounded-2xl border bg-card text-sm text-muted-foreground">
              No API example available for this model ability.
            </div>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
