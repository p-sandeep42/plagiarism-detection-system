"use client";

export default function SplitView({ sourceText, targetText, highlights }: any) {
  const renderHighlightedText = (text: string, isSource: boolean) => {
    if (!text) return "No content";
    if (!highlights || highlights.length === 0) return text;
    
    let lastIndex = 0;
    const elements = [];
    
    // Sort highlights by start index to render linearly
    const sortedHighlights = [...highlights].sort((a, b) => 
      isSource ? a.source_index_start - b.source_index_start : a.target_index_start - b.target_index_start
    );

    sortedHighlights.forEach((hl: any, idx: number) => {
      const start = isSource ? hl.source_index_start : hl.target_index_start;
      const end = isSource ? hl.source_index_end : hl.target_index_end;
      
      if (start > lastIndex) {
        elements.push(<span key={`text-${idx}`}>{text.substring(lastIndex, start)}</span>);
      }
      
      if (start >= lastIndex) {
        // Dynamic styling based on match type
        const isExact = hl.match_type === "exact";
        const colorClass = isExact
            ? "bg-blue-500/30 text-blue-100 border-b-2 border-blue-500 hover:bg-blue-500/50"
            : hl.match_type === "structural"
                ? "bg-emerald-500/30 text-emerald-100 border-b-2 border-emerald-500 hover:bg-emerald-500/50"
                : "bg-purple-500/30 text-purple-100 border-b-2 border-purple-500 hover:bg-purple-500/50";

        elements.push(
          <span 
            key={`hl-${idx}`} 
            className={`${colorClass} cursor-pointer rounded-sm px-1 transition-colors`} 
            title={`Match type: ${hl.match_type}`}
          >
            {text.substring(start, end)}
          </span>
        );
        lastIndex = end;
      }
    });

    if (lastIndex < text.length) {
      elements.push(<span key="text-end">{text.substring(lastIndex)}</span>);
    }

    return elements;
  };

  return (
    <div className="w-full flex gap-4 mt-8 relative">
      <div className="flex-1 glass-panel p-6 h-[600px] overflow-y-auto font-mono text-sm leading-relaxed whitespace-pre-wrap">
        <h3 className="font-bold text-aura-accent mb-4 border-b border-aura-border pb-2">Source Document</h3>
        <p className="text-gray-300 relative z-20">
          {renderHighlightedText(sourceText, true)}
        </p>
      </div>
      
      {/* SVG Canvas for arrows between views - Placeholder for logic */}
      <svg className="absolute inset-0 pointer-events-none w-full h-full z-10" />

      <div className="flex-1 glass-panel p-6 h-[600px] overflow-y-auto font-mono text-sm leading-relaxed whitespace-pre-wrap">
        <h3 className="font-bold text-aura-danger mb-4 border-b border-aura-border pb-2">Target Document</h3>
        <p className="text-gray-300 relative z-20">
          {renderHighlightedText(targetText, false)}
        </p>
      </div>
    </div>
  );
}
