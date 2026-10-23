import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Renders model output as Markdown. Raw HTML is never rendered (react-markdown's default). */
export function Markdown({ text }: { text: string }) {
  return (
    <div className="prose-answer">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}
