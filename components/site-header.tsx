"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LogoIcon } from "./icons";
import { ServerStatusPill } from "./server-status";

const NAV_ITEMS = [
  { href: "/", label: "ホーム" },
  { href: "/train", label: "モデルを作る" },
  { href: "/convert", label: "声を変換" },
  { href: "/models", label: "モデル一覧" },
] as const;

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="border-b border-zinc-200 bg-white/80 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/80">
      <div className="mx-auto flex w-full max-w-5xl flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3 sm:px-6">
        <Link href="/" className="flex items-center gap-2 font-bold tracking-tight">
          <span className="flex size-8 items-center justify-center rounded-lg bg-violet-600 text-white dark:bg-violet-500 dark:text-zinc-950">
            <LogoIcon />
          </span>
          RVC Studio
        </Link>
        <nav aria-label="メインメニュー" className="order-3 w-full sm:order-none sm:w-auto">
          <ul className="flex flex-wrap gap-1">
            {NAV_ITEMS.map((item) => {
              const active =
                item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    aria-current={active ? "page" : undefined}
                    className={`block rounded-lg px-3 py-1.5 text-sm transition-colors ${
                      active
                        ? "bg-violet-50 font-medium text-violet-700 dark:bg-violet-500/15 dark:text-violet-300"
                        : "text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800"
                    }`}
                  >
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
        <div className="ml-auto">
          <ServerStatusPill />
        </div>
      </div>
    </header>
  );
}
