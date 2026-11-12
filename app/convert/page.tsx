import type { Metadata } from "next";
import { Suspense } from "react";
import { PageHeader } from "@/components/ui";
import { ConvertWorkspace } from "./convert-workspace";

export const metadata: Metadata = { title: "声を変換" };

export default function ConvertPage() {
  return (
    <>
      <PageHeader
        title="声を変換する"
        description="歌や話し声の音声ファイルを、作ったモデルの声に変換します。"
      />
      {/* URL の ?model= を読むため、Suspense で囲む */}
      <Suspense fallback={null}>
        <ConvertWorkspace />
      </Suspense>
    </>
  );
}
