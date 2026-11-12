"use client";

import { api } from "@/lib/api";
import { usePolling } from "@/lib/hooks";
import { ButtonLink, Card, EmptyState, SectionTitle } from "@/components/ui";
import { ImportForm } from "./import-form";
import { ModelCard } from "./model-card";

export function ModelsWorkspace() {
  const { data: models, refresh } = usePolling(api.models, 5000);

  return (
    <div className="space-y-8">
      <section aria-labelledby="model-list">
        <SectionTitle
          action={
            <ButtonLink href="/train" size="sm" variant="secondary">
              新しく作る
            </ButtonLink>
          }
        >
          <span id="model-list">モデル {models ? `（${models.length}）` : ""}</span>
        </SectionTitle>
        {models === undefined ? (
          <EmptyState>読み込み中…</EmptyState>
        ) : models.length === 0 ? (
          <EmptyState>
            まだモデルがありません。「新しく作る」から学習するか、下のフォームで取り込んでください。
          </EmptyState>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {models.map((model) => (
              <ModelCard key={model.name} model={model} onChanged={() => void refresh()} />
            ))}
          </div>
        )}
      </section>

      <Card aria-labelledby="import-model">
        <SectionTitle>
          <span id="import-model">他で作ったモデルを取り込む</span>
        </SectionTitle>
        <ImportForm onImported={() => void refresh()} />
      </Card>
    </div>
  );
}
