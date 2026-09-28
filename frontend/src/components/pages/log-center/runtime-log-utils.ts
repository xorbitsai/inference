export function tailLogLines(logs: string, maxLines: number): string {
  let end = logs.endsWith('\n') ? logs.length - 1 : logs.length;

  for (let line = 0; line < maxLines; line++) {
    if (end <= 0) return logs;
    const newline = logs.lastIndexOf('\n', end - 1);
    if (newline < 0) return logs;
    end = newline;
  }

  return logs.slice(end + 1);
}
