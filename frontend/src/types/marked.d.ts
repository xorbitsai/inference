declare module 'marked' {
  export interface MarkedOptions {
    async?: boolean;
    breaks?: boolean;
    gfm?: boolean;
    headerIds?: boolean;
    mangle?: boolean;
    sanitize?: boolean;
    sanitizer?: (html: string) => string;
    smartLists?: boolean;
    smartypants?: boolean;
    xhtml?: boolean;
  }

  export interface Renderer {
    code(code: string, language: string): string;
    blockquote(quote: string): string;
    html(html: string): string;
    heading(text: string, level: number): string;
    hr(): string;
    list(body: string, ordered: boolean): string;
    listitem(text: string): string;
    paragraph(text: string): string;
    table(header: string, body: string): string;
    tablerow(content: string): string;
    tablecell(content: string, flags: { header: boolean; align: 'center' | 'left' | 'right' | null }): string;
    strong(text: string): string;
    em(text: string): string;
    codespan(code: string): string;
    br(): string;
    del(text: string): string;
    link(href: string, title: string, text: string): string;
    image(href: string, title: string, text: string): string;
    text(text: string): string;
  }

  export class Lexer {
    constructor(options?: MarkedOptions);
    lex(src: string): Token[];
  }

  export interface Token {
    type: string;
    raw: string;
    [key: string]: any;
  }

  export class Parser {
    constructor(options?: MarkedOptions);
    parse(tokens: Token[]): string;
  }

  export function parse(src: string, options?: MarkedOptions): string;
  export function parseAsync(src: string, options?: MarkedOptions): Promise<string>;
  export function setOptions(options: MarkedOptions): void;
  export function use(extension: any): void;
  export function walkTokens(tokens: Token[], callback: (token: Token) => void): void;
  export function getDefaults(): MarkedOptions;
  export function defaults(options: MarkedOptions): MarkedOptions;

  export const marked: {
    parse: typeof parse;
    parseAsync: typeof parseAsync;
    setOptions: typeof setOptions;
    use: typeof use;
    walkTokens: typeof walkTokens;
    getDefaults: typeof getDefaults;
    defaults: typeof defaults;
    Lexer: typeof Lexer;
    Parser: typeof Parser;
    Renderer: typeof Renderer;
    options: MarkedOptions;
  };
}
