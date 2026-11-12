import type { Metadata } from "next";
import { PageHeader } from "@/components/ui";
import { TrainWorkspace } from "./train-workspace";

export const metadata: Metadata = { title: "モデルを作る" };

export default function TrainPage() {
  return (
    <>
      <PageHeader
        title="声のモデルを作る"
        description="録音した声を学習させて、その声に変換できるモデルを作ります。学習はこの PC の中で行い、音声が外に送られることはありません。"
      />
      <TrainWorkspace />
    </>
  );
}
