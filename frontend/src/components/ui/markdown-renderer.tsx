import { FC, PropsWithChildren } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { cn } from '@/lib/utils';

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
      // rehypeKatex runs AFTER rehypeSanitize: raw HTML from the model/user
      // (via rehypeRaw) is sanitized with the plain default schema first, so
      // it can never smuggle in a "style"/"className" via e.g. a raw <span>.
      // KaTeX's own HTML (span/math/... with "katex*" classes and inline
      // "style") is generated afterwards, so it's never re-sanitized and
      // always renders correctly.
      rehypePlugins={
        parseHtml
          ? [rehypeRaw, [rehypeSanitize, defaultSchema], rehypeKatex]
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
