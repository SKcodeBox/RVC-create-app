import type { Metadata } from "next";
import { PageHeader } from "@/components/ui";
import { ModelsWorkspace } from "./models-workspace";

export const metadata: Metadata = { title: "モデル一覧" };

export default function ModelsPage() {
  return (
    <>
      <PageHeader
        title="モデル一覧"
        description="作ったモデルと、取り込んだモデルです。学習の続きや、モデルファイルの保存もここからできます。"
      />
      <ModelsWorkspace />
    </>
  );
}
