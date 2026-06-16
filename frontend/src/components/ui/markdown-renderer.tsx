import { FC, PropsWithChildren } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';
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
      remarkPlugins={[remarkGfm]}
      rehypePlugins={parseHtml ? [rehypeRaw, rehypeSanitize] : []}
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
