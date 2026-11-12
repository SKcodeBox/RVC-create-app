import Link from "next/link";
import type { ReactNode } from "react";
import { ArrowRightIcon, MicIcon, WaveIcon } from "@/components/icons";
import { HomeStatus } from "./home-status";

export default function HomePage() {
  return (
    <div className="space-y-10">
      <section>
        <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">RVC Studio</h1>
        <p className="mt-3 max-w-2xl text-lg text-zinc-600 dark:text-zinc-400">
          声の録音から自分だけの声のモデルを作り、歌や話し声をその声に変換できます。
          処理はすべてこの PC（Windows・Mac）の中で行います。
        </p>
      </section>

      <div className="grid gap-4 sm:grid-cols-2">
        <ActionCard
          href="/train"
          icon={<MicIcon width={24} height={24} />}
          title="声のモデルを作る"
          description="声の録音（10〜30分がおすすめ）を学習させて、その声のモデルを作ります。"
        />
        <ActionCard
          href="/convert"
          icon={<WaveIcon width={24} height={24} />}
          title="声を変換する"
          description="音声ファイルを、作ったモデルの声に変換します。声の高さも調整できます。"
        />
      </div>

      <HomeStatus />

      <section aria-labelledby="how-to">
        <h2 id="how-to" className="mb-4 text-lg font-semibold">
          使い方
        </h2>
        <ol className="grid gap-4 sm:grid-cols-3">
          <Step number={1} title="声を用意する">
            声だけが入った音声を用意します。BGM や雑音、他の人の声が入っていないものが向いています。
          </Step>
          <Step number={2} title="モデルを作る">
            音声を選んで学習を始めます。数時間かかることもありますが、ほかのページを見ていても学習は続きます。
          </Step>
          <Step number={3} title="変換する">
            変換したい音声とモデルを選びます。男性→女性なら +12 など、声の高さを合わせるのがコツです。
          </Step>
        </ol>
      </section>
    </div>
  );
}

function ActionCard({
  href,
  icon,
  title,
  description,
}: {
  href: string;
  icon: ReactNode;
  title: string;
  description: string;
}) {
  return (
    <Link
      href={href}
      className="group flex flex-col gap-3 rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm transition hover:border-violet-400 hover:shadow-md dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-violet-500"
    >
      <span className="flex size-11 items-center justify-center rounded-xl bg-violet-100 text-violet-700 dark:bg-violet-500/15 dark:text-violet-300">
        {icon}
      </span>
      <span className="flex items-center gap-2 text-lg font-semibold">
        {title}
        <ArrowRightIcon
          width={18}
          height={18}
          className="text-zinc-400 transition group-hover:translate-x-0.5 group-hover:text-violet-600 dark:group-hover:text-violet-400"
        />
      </span>
      <span className="text-sm text-zinc-600 dark:text-zinc-400">{description}</span>
    </Link>
  );
}

function Step({
  number,
  title,
  children,
}: {
  number: number;
  title: string;
  children: ReactNode;
}) {
  return (
    <li className="rounded-2xl border border-zinc-200 p-5 dark:border-zinc-800">
      <span className="flex size-7 items-center justify-center rounded-full bg-zinc-900 text-sm font-semibold text-white dark:bg-zinc-100 dark:text-zinc-900">
        {number}
      </span>
      <p className="mt-3 font-semibold">{title}</p>
      <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">{children}</p>
    </li>
  );
}
