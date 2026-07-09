import { FC, PropsWithChildren } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { cn } from '@/lib/utils';

// rehype-sanitize's GitHub-derived default schema doesn't know about the
// classes/attributes rehype-katex's HTML output needs (span/math/semantics/...
// with "katex*" classes and inline "style"), so it would strip them back out.
// Extend the schema rather than replace it, so GFM sanitation elsewhere is
// unaffected.
const katexSchema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    span: [...(defaultSchema.attributes?.span || []), 'className', 'style', 'ariaHidden'],
    math: ['xmlns'],
    annotation: ['encoding'],
  },
  tagNames: [
    ...(defaultSchema.tagNames || []),
    'math',
    'semantics',
    'mrow',
    'mi',
    'mn',
    'mo',
    'msup',
    'msub',
    'msubsup',
    'mfrac',
    'msqrt',
    'mroot',
    'mtable',
    'mtr',
    'mtd',
    'mover',
    'munder',
    'munderover',
    'mtext',
    'mspace',
    'mstyle',
    'mpadded',
    'menclose',
    'annotation',
  ],
};

interface ReactMarkdownProps {
  classnames?: string;
  parseHtml?: boolean;
}
const ReactMarkdown: FC<PropsWithChildren<ReactMarkdownProps>> = ({
  children,
  classnames: classname,
  parseHtml = false,
}) => {
  if (typeof children !== 'string') {
    return <span>{String(children)}</span>;
  }
  return (
    <Markdown
      skipHtml={false}
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={
        parseHtml
          ? [rehypeRaw, rehypeKatex, [rehypeSanitize, katexSchema]]
          : [rehypeKatex]
      }
      className={cn('markdown-body break-word', classname)}
      components={{
        a: (props) => {
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          const { node, ...rest } = props;
          return <a {...rest} target="_blank" />;
        },
      }}
    >
      {children}
    </Markdown>
  );
};
export default ReactMarkdown;
