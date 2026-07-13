'use client';

import { FC, useRef, useState } from 'react';
import {
  Paperclip,
  Send,
  Wrench,
  BrushCleaning,
  Bot,
  User,
  Loader2,
  ChevronUp,
  Copy,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { FileUpload, FileUploadResult } from '@/components/ui/file-upload';
import { Form } from '@/components/ui/form';
import { FormField } from '@/components/ui/form-field';
import { Input } from '@/components/ui/input';
import { MediaPreview } from '@/components/ui/media-preview';
import { Switch } from '@/components/ui/switch';
import {
  Dialog,
  DialogHeader,
  DialogContent,
  DialogFooter,
  DialogTitle,
} from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import { InfoTooltip } from '@/components/ui/tooltip';
import ReactMarkdown from '@/components/ui/markdown-renderer';
import { useForm } from '@/hooks/use-form';
import { ModelAbility } from '@/constants';
import {
  postEventStreamFetcher,
  PostEventStreamFetcherOptions,
  EventStreamController,
} from '@/lib/eventStream';
import { cn, sleep, copyToClipboard } from '@/lib/utils';
import { isNumber } from '@/lib/is';
import type { RunningModelDetail, ChatStreamResult, ChatChoicesMessage } from '@/types/services';
import type { FormValues } from '@/types/form';
import type { FileUploadValue } from '@/types/common';

import type { ChatMessage, ChatSettings } from '../types';
import { transformFileInfoForResult } from '../utils';

const fileDataUrlCache = new WeakMap<File, string>();

function fileToDataURL(file: File): Promise<string> {
  const cached = fileDataUrlCache.get(file);
  if (cached) return Promise.resolve(cached);

  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      const result = String(reader.result || '');
      fileDataUrlCache.set(file, result);
      resolve(result);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

const ChatItem: FC<{ data: ChatMessage }> = ({ data }) => {
  const { role, content, loading, success, attachment, thinkingContent, thinkingCompleted, usage } =
    data;
  const [thinkingOpen, setThinkingOpen] = useState(true);

  const renderFileInfo = () => {
    if (!attachment) return null;
    if (role === 'user' && attachment.file) {
      return <FileUploadResult fileList={[attachment as FileUploadValue]} />;
    }

    return <MediaPreview type={attachment.type} url={attachment.url} />;
  };
  if (role === 'user') {
    return (
      <div className="flex w-full justify-end gap-2 fade-in">
        <div className="flex min-w-0 max-w-[78%] flex-col items-end gap-2">
          {renderFileInfo()}

          {!!content && (
            <div className="max-w-full break-words rounded-lg rounded-tr-none bg-primary p-2 text-[#fff]">
              {content}
            </div>
          )}
        </div>

        <div className="flex size-9 shrink-0 items-center justify-center rounded-full bg-muted">
          <User className="size-5" />
        </div>
      </div>
    );
  }

  const renderThinking = () => {
    if (!thinkingContent) return null;
    return (
      <div>
        <div
          className="w-fit flex items-center justify-center gap-1 p-2 bg-muted/50 rounded-lg cursor-pointer hover:bg-muted"
          onClick={() => setThinkingOpen(!thinkingOpen)}
        >
          {!thinkingCompleted && <Loader2 className="size-4 animate-spin text-muted-foreground" />}
          <span>{thinkingCompleted ? 'Thinking Completed' : 'Thinking...'}</span>
          <ChevronUp
            className={cn(
              'size-4 ml-1 text-xs transition-out rotate-180 transition-transform duration-300 ease-out',
              thinkingOpen && 'rotate-0'
            )}
          />
        </div>
        {thinkingOpen && (
          <div className="flex gap-4 mt-2 text-muted-foreground break-word">
            <div className="border-l-0.5 border border-border/50" />
            <ReactMarkdown>{thinkingContent}</ReactMarkdown>
          </div>
        )}
      </div>
    );
  };
  const renderAnswerFooter = () => {
    if (!success) return null;
    const tokens =
      isNumber(usage?.prompt_tokens) &&
      isNumber(usage?.completion_tokens) &&
      isNumber(usage?.total_tokens)
        ? `${usage?.prompt_tokens} → ${usage?.completion_tokens} (∑ ${usage?.total_tokens})`
        : '';
    if (usage?.completion_tokens === 0) {
      return <div>{tokens}</div>;
    }
    return (
      <div className="pt-2 border-t flex items-center justify-between gap-2 text-muted-foreground">
        <div>{!!(usage?.total_tokens && usage.total_tokens > 0) && tokens}</div>
        <Copy
          className="size-4 cursor-pointer hover:text-foreground"
          onClick={() => copyToClipboard(content)}
        />
      </div>
    );
  };
  return (
    <div className="flex w-full justify-start gap-2">
      <div className="flex size-9 shrink-0 items-center justify-center rounded-full bg-muted">
        <Bot className="size-5" />
      </div>
      <div className="flex min-w-0 max-w-[78%] flex-col gap-2">
        {renderThinking()}
        <div className="w-fit max-w-full rounded-lg rounded-tl-none bg-muted p-2">
          {loading ? (
            <Loader2 className="size-5 animate-spin text-muted-foreground" />
          ) : (
            <div className="flex max-w-full flex-col gap-2 break-words">
              {!!content && <ReactMarkdown>{content}</ReactMarkdown>}
              {renderFileInfo()}
              {renderAnswerFooter()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
interface ChatPanelProps {
  model: RunningModelDetail;
  modelUid: string;
}

export function ChatPanel({ model, modelUid }: ChatPanelProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const [form] = useForm();
  const [settingOpen, setSettingOpen] = useState(false);
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<FileUploadValue[]>([]);
  const [loading, setLoading] = useState(false);
  const [chatController, setChatController] = useState<EventStreamController | null>(null);
  const [chatList, setChatList] = useState<ChatMessage[]>([]);
  const [settings, setSettings] = useState<ChatSettings>({
    max_tokens: 0,
    temperature: 1,
    stream: true,
  });

  const hasVision = model.model_ability.includes(ModelAbility.Vision);
  const hasAudio = model.model_ability.includes(ModelAbility.Audio);
  const accept = [hasVision ? 'image/*,video/*' : '', hasAudio ? 'audio/*' : '']
    .filter(Boolean)
    .join(',');
  const handleSaveSetting = (values: FormValues) => {
    setSettings(values as ChatSettings);
    setSettingOpen(false);
  };

  const onData = async (chunk: ChatStreamResult) => {
    const content = settings.stream
      ? chunk?.choices?.[0]?.delta?.content || ''
      : chunk?.choices?.[0]?.message?.content || '';
    const thinkingContent = chunk?.choices?.[0]?.delta?.reasoning_content;
    setChatList((prevList) => {
      const lastChatItem = prevList[prevList.length - 1];
      if (!lastChatItem) {
        return prevList;
      }
      // A null reasoning_content field means the reasoning phase has completed.
      const thinkingCompleted =
        'reasoning_content' in (chunk?.choices?.[0]?.delta || {}) ? thinkingContent === null : true;
      const updatedLastItem = {
        ...lastChatItem,
        usage: chunk?.usage,
        content: lastChatItem.content + content,
        // If in thinking, the content below is in loading state and the status is true, otherwise it is false
        loading: !thinkingCompleted,
        success: true,
        thinkingContent: (lastChatItem.thinkingContent || '') + (thinkingContent || ''),
        thinkingCompleted,
        attachment:
          transformFileInfoForResult(chunk?.choices?.[0]?.message as ChatChoicesMessage) ||
          lastChatItem.attachment,
      };
      return [...prevList.slice(0, -1), updatedLastItem];
    });
    await sleep(10);
    bottomRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'end',
    });
  };
  const onError: PostEventStreamFetcherOptions<ChatStreamResult>['onError'] = (msg) => {
    setChatList((prevList) => {
      return [
        ...prevList.slice(0, -1),
        {
          ...prevList[prevList.length - 1],
          content: msg,
          // If there is an error, hide the thought process and only display the reason for the error
          thinkingContent: '',
          thinkingCompleted: true,
          loading: false,
          success: false,
        },
      ];
    });
  };
  const onEnd = () => {
    setLoading(false);
    setChatController(null);
  };
  const send = async (value: string = '') => {
    const text = value.trim();
    if (!text || loading) return;

    const newChatList: ChatMessage[] = [
      ...chatList,
      {
        role: 'user',
        content: value,
        loading: false,
        ...(attachments.length ? { attachment: attachments[0] } : {}),
      },
    ];
    setChatList(newChatList);
    await sleep(10);
    bottomRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'end',
    });
    setInput('');

    const newController = new EventStreamController();
    setLoading(true);
    setAttachments([]);
    setChatController(newController);
    setChatList([
      ...newChatList,
      {
        role: 'assistant',
        content: '',
        loading: true,
        thinkingCompleted: false,
      },
    ]);
    await sleep(10);
    bottomRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'end',
    });

    const messages = await Promise.all(
      newChatList.map(async (item) => {
        if (item.role === 'user' && item.attachment) {
          const { type, url, file } = item.attachment || {};
          const attachmentUrl = file ? await fileToDataURL(file) : url;
          return {
            role: item.role,
            content: [
              { type: 'text', text: item.content },
              {
                type: `${type}_url`,
                [`${type}_url`]: { url: attachmentUrl },
              },
            ],
          };
        }
        return { role: item.role, content: item.content };
      })
    );

    await postEventStreamFetcher<ChatStreamResult>(
      {
        url: '/v1/chat/completions',
        data: {
          model: modelUid,
          ...settings,
          // if max_tokens = 0, fetch err
          max_tokens: settings.max_tokens || undefined,
          messages,
          stream_options: { include_usage: true },
        },
        options: {
          onData,
          onError,
          onEnd,
        },
      },
      newController
    );
  };
  const handleEnter = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.shiftKey && event.key === 'Enter') {
      return;
    }
    if (event.key === 'Enter') {
      event.preventDefault();
      send(event.currentTarget.value);
    }
  };
  const handleCancel = () => {
    if (!chatController) return;
    // cancel fetch
    chatController.terminate();
    setChatList((prevList) => {
      return [
        ...prevList.slice(0, -1),
        {
          ...prevList[prevList.length - 1],
          thinkingCompleted: true,
          success: true,
          loading: false,
        },
      ];
    });
    setChatController(null);
    setLoading(false);
  };
  const handleClear = () => {
    if (loading && chatController) {
      chatController.terminate();
    }
    setChatList([]);
  };

  return (
    <>
      <div className="flex h-[calc(100vh-216px)] flex-col overflow-hidden rounded-xl border bg-card shadow-sm">
        <div className="min-h-0 flex-1 overflow-y-auto bg-background/40 px-6 py-6">
          {!chatList.length && (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              Start by sending a message or attaching media.
            </div>
          )}
          <div className="flex flex-col gap-4">
            {chatList.map((chat, index) => (
              <ChatItem key={index} data={chat} />
            ))}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="border-t bg-card p-4 pb-2 flex flex-col gap-2">
          {!!attachments.length && (
            <FileUploadResult
              className="w-1/3"
              fileList={attachments}
              onRemove={(index) =>
                setAttachments((current) =>
                  current.filter((_, currentIndex) => currentIndex !== index)
                )
              }
            />
          )}
          <Textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Please enter"
            className="min-h-20 flex-1 border-0 bg-muted/60 text-base shadow-none focus-visible:ring-0 rounded-xl"
            onKeyDown={handleEnter}
          />
          <div className="flex justify-between items-center gap-2">
            <div className="flex items-center">
              <InfoTooltip content="Additional Inputs">
                <Button
                  variant="outline"
                  size="icon"
                  className="size-8 rounded-full border-0"
                  onClick={() => setSettingOpen(true)}
                >
                  <Wrench />
                </Button>
              </InfoTooltip>

              {!!accept && (
                <FileUpload
                  value={attachments}
                  onChange={setAttachments}
                  accept={accept}
                  showResult={false}
                >
                  <Button variant="outline" size="icon" className="size-8 rounded-full border-0">
                    <Paperclip />
                  </Button>
                </FileUpload>
              )}
              <Button
                variant="outline"
                size="icon"
                className="size-8 rounded-full border-0"
                onClick={handleClear}
              >
                <BrushCleaning className="rotate-45" />
              </Button>
            </div>
            {!!chatController ? (
              <Button size="icon" className="size-8 rounded-full" onClick={handleCancel}>
                <div className="bg-[#fff] size-3 rounded-[2px]" />
              </Button>
            ) : (
              <Button
                size="icon"
                className="size-8 rounded-full"
                loading={loading}
                onClick={() => send(input)}
              >
                <Send />
              </Button>
            )}
          </div>
        </div>
      </div>
      <Dialog open={settingOpen} onOpenChange={setSettingOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Additional Inputs</DialogTitle>
          </DialogHeader>
          <Form form={form} onFinish={handleSaveSetting} initialValues={{ ...settings }}>
            <FormField
              name="max_tokens"
              label="Max Tokens (0 stands for maximum possible tokens)"
              normalize={(value) => Number(value)}
            >
              <Input type="number" min={0} max={model.context_length || undefined} />
            </FormField>
            <FormField name="temperature" label="Temperature" normalize={(value) => Number(value)}>
              <Input type="number" min={0} max={2} step={0.01} />
            </FormField>
            <FormField name="stream" label="Stream" valuePropName="checked">
              <Switch />
            </FormField>
            <DialogFooter>
              <Button type="submit">Save</Button>
            </DialogFooter>
          </Form>
        </DialogContent>
      </Dialog>
    </>
  );
}
